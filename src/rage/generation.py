"""Fixed-length greedy decoding with the official T5 forward and cache classes.

A CUDA graph removes repeated Python/kernel-launch overhead. Every replay
recomputes the encoder and every decoder step with the current model weights;
no model-dependent states are reused across calls or optimizer updates.
"""
from __future__ import annotations

import torch


class GreedyGraph:
    def __init__(self, model, input_ids, attention_mask, horizon, *, output_scores=False):
        from transformers.cache_utils import StaticCache, EncoderDecoderCache

        self.ids = input_ids.clone()
        self.mask = attention_mask.clone()
        batch, context = input_ids.shape
        self.tokens = torch.empty(batch, horizon + 1, dtype=torch.long, device=input_ids.device)
        self.scores = (torch.empty(batch, horizon, model.config.vocab_size,
                                  dtype=torch.float32, device=input_ids.device)
                       if output_scores else None)
        config = model.generation_config
        start_token = config.decoder_start_token_id
        eos = config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        eos_indices = torch.tensor(eos, dtype=torch.long, device=input_ids.device)
        positions = [torch.tensor([i], device=input_ids.device) for i in range(horizon)]
        decoder_mask = torch.ones(batch, horizon, dtype=torch.long, device=input_ids.device)
        cache = EncoderDecoderCache(
            StaticCache(config=model.config, max_cache_len=horizon),
            StaticCache(config=model.config, max_cache_len=context))

        def forward():
            cache.reset()
            self.tokens[:, 0] = start_token
            encoded = model.get_encoder()(input_ids=self.ids, attention_mask=self.mask,
                                          return_dict=True)
            for step in range(horizon):
                output = model(
                    encoder_outputs=encoded, attention_mask=self.mask,
                    decoder_input_ids=self.tokens[:, step:step + 1],
                    decoder_attention_mask=decoder_mask,
                    past_key_values=cache, cache_position=positions[step],
                    use_cache=True, return_dict=True)
                logits = output.logits[:, -1, :].float().clone()
                # min_new_tokens == max_new_tokens == horizon, so EOS remains
                # suppressed for every one of the horizon decoding steps.
                logits.index_fill_(-1, eos_indices, -torch.inf)
                if self.scores is not None:
                    self.scores[:, step] = logits
                self.tokens[:, step + 1] = logits.argmax(-1)

        stream = torch.cuda.Stream(device=input_ids.device)
        stream.wait_stream(torch.cuda.current_stream(input_ids.device))
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                forward()
        torch.cuda.current_stream(input_ids.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            forward()
        self.cache = cache
        # Graph replay uses these input addresses without owning Python refs.
        self._constants = (positions, decoder_mask, eos_indices)

    def __call__(self, input_ids, attention_mask):
        self.ids.copy_(input_ids)
        self.mask.copy_(attention_mask)
        self.graph.replay()
        return self.tokens[:, 1:].clone()


def supports_greedy_graph(model):
    """Only the fixed-length Chronos/T5 decoding contract is captured here.

    Other generation presets keep the official generate implementation. These
    checks run once per model, not at every token or block.
    """
    if getattr(model.config, "model_type", None) != "t5":
        return False
    config = model.generation_config
    if config.decoder_start_token_id is None:
        return False
    neutral = {"repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0,
               "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0,
               "diversity_penalty": 0.0}
    options = ("bad_words_ids", "force_words_ids", "forced_bos_token_id", "forced_eos_token_id",
               "constraints", "exponential_decay_length_penalty", "suppress_tokens",
               "begin_suppress_tokens", "sequence_bias", "watermarking_config", "token_healing",
               "remove_invalid_values", "renormalize_logits")
    return (all(getattr(config, key, value) == value for key, value in neutral.items())
            and not any(getattr(config, key, None) for key in options))


@torch.no_grad()
def generate_tokens(engine, input_ids, attention_mask, *, output_scores=False):
    """Common forecast backend for selection and downstream evaluation."""
    from rage.execution import execution_policy

    horizon = engine.config.prediction_length
    use_graph = (execution_policy(engine).generation == "cuda_graph"
                 and input_ids.device.type == "cuda")
    if use_graph:
        supported = getattr(engine, "_greedy_graph_supported", None)
        if supported is None:
            supported = supports_greedy_graph(engine.model)
            engine._greedy_graph_supported = supported
        use_graph = supported
    if use_graph:
        graphs = getattr(engine, "_greedy_graphs", None)
        if graphs is None:
            graphs = engine._greedy_graphs = {}
        key = (tuple(input_ids.shape), horizon, output_scores)
        if key not in graphs:
            graphs[key] = GreedyGraph(engine.model, input_ids, attention_mask, horizon,
                                     output_scores=output_scores)
        graph = graphs[key]
        tokens = graph(input_ids, attention_mask)
        return tokens, graph.scores.clone() if output_scores else None
    result = engine.model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        min_new_tokens=horizon, max_new_tokens=horizon, do_sample=False,
        num_beams=1, num_return_sequences=1,
        **({"return_dict_in_generate": True, "output_scores": True} if output_scores else {}))
    if output_scores:
        return result.sequences[:, 1:], torch.stack(result.scores, dim=1).float()
    return result[:, 1:], None
