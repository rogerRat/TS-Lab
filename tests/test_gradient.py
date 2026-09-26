"""CPU contracts for the root-level gradient module."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gradient import ModelNotFoundError
from gradient.chronos import ChronosTinyConfig, ChronosTinyGradientEngine, select_parameter
from gradient.projection import ParameterHashProjector, TensorSketchProjector
from gradient.warmup import WarmupCandidateRunner


class _Tokenizer:
    def context_input_transform(self, contexts: torch.Tensor):
        batch = contexts.shape[0]
        ids = torch.arange(3).repeat(batch, 1) % 5
        return ids.long(), torch.ones_like(ids), torch.ones(batch, 1)

    def label_input_transform(self, futures: torch.Tensor, _scale: torch.Tensor):
        labels = futures.round().long().remainder(5)
        return labels, torch.ones_like(labels)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(5, 3)
        self.lm_head = nn.Linear(3, 5, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        hidden = self.emb(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="mean")
        return SimpleNamespace(logits=logits, loss=loss)


def _engine(layer="lm_head.weight", projection_dimension=17):
    torch.manual_seed(7)
    config = ChronosTinyConfig(
        context_length=2,
        prediction_length=3,
        n_tokens=5,
        layer=layer,
        projection_dimension=projection_dimension,
        projection_repeats=2,
        token_chunk_size=1,
        gradient_batch_size=2,
        warmup_steps=2,
        warmup_batch_size=2,
        warmup_learning_rates=(0.01, 0.02),
    )
    return ChronosTinyGradientEngine(config, device="cpu", model=_ToyModel(), tokenizer=_Tokenizer())


def test_tensor_sketch_is_deterministic_and_chunked() -> None:
    first = TensorSketchProjector(5, 3, projection_dimension=13, repeats=2, seed=4, token_chunk_size=1)
    second = TensorSketchProjector(5, 3, projection_dimension=13, repeats=2, seed=4, token_chunk_size=3)
    logits_gradient = torch.arange(2 * 5 * 5, dtype=torch.float32).reshape(2, 5, 5) / 11
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3) / 7
    left = first.project(logits_gradient, hidden)
    right = second.project(logits_gradient, hidden)
    assert left.shape == (2, 13)
    assert torch.isfinite(left).all()
    torch.testing.assert_close(left, right)


def test_parameter_hash_projector_is_fixed_shape_and_seeded() -> None:
    gradient = torch.arange(23, dtype=torch.float32)
    first = ParameterHashProjector(23, projection_dimension=9, seed=10, value_chunk_size=4)
    repeat = ParameterHashProjector(23, projection_dimension=9, seed=10, value_chunk_size=4)
    assert first.project(gradient).shape == (9,)
    torch.testing.assert_close(first.project(gradient), repeat.project(gradient))
    with pytest.raises(ValueError, match="size"):
        first.project(torch.ones(22))


def test_full_projection_compact_tables_preserve_original_global_random_stream():
    from rags.engines import _FullModelProjector
    params = [("a", torch.nn.Parameter(torch.ones(7))), ("b", torch.nn.Parameter(torch.ones(16)))]
    projector = _FullModelProjector(params, 9, 10, torch.device("cpu"))
    rng = np.random.default_rng(10)
    hashes = torch.from_numpy(rng.integers(0, 9, size=23))
    signs = torch.from_numpy(rng.choice((-1., 1.), size=23)).float()
    assert torch.equal(torch.cat([h.hashes for h in projector.hashers]).long(), hashes)
    assert torch.equal(torch.cat([h.signs for h in projector.hashers]).float(), signs)
    expected = torch.zeros(9).index_add_(0, hashes, signs * torch.arange(23).float())
    actual = projector.project([torch.arange(7).float(), torch.arange(7, 23).float()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sum(h.hashes.nbytes + h.signs.nbytes for h in projector.hashers) == 23 * 5


def test_default_lm_head_uses_fast_path_and_is_batch_partition_invariant() -> None:
    engine = _engine()
    values = np.asarray([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 1, 2, 2, 3]], dtype=np.float32)
    losses_a, sketches_a = engine.batch_gradient_sketch(values, batch_size=2)
    losses_b, sketches_b = engine.batch_gradient_sketch(values, batch_size=1)
    assert engine.layer_name == "lm_head.weight"
    assert engine.fast_lm_head
    assert losses_a.shape == (3,)
    assert sketches_a.shape == (3, 17)
    assert np.isfinite(sketches_a).all()
    np.testing.assert_allclose(losses_a, losses_b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sketches_a, sketches_b, rtol=1e-5, atol=1e-6)


def test_explicit_arbitrary_layer_uses_one_sample_projection() -> None:
    engine = _engine(layer="emb.weight", projection_dimension=11)
    values = np.asarray([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=np.float32)
    losses, sketches = engine.batch_gradient_sketch(values)
    assert engine.layer_name == "emb.weight"
    assert not engine.fast_lm_head
    assert losses.shape == (2,)
    assert sketches.shape == (2, 11)
    assert np.isfinite(sketches).all()


def test_layer_alias_and_regex_selection() -> None:
    model = _ToyModel()
    assert select_parameter(model, "default")[0] == "lm_head.weight"
    assert select_parameter(model, "re:emb\\.weight")[0] == "emb.weight"
    with pytest.raises(ValueError, match="not found"):
        select_parameter(model, "missing.weight")


def test_default_layer_survives_tied_t5_style_named_parameter_deduplication() -> None:
    model = _ToyModel()
    model.lm_head.weight = model.emb.weight
    name, parameter = select_parameter(model, "default")
    assert name == "lm_head.weight"
    assert parameter is model.lm_head.weight


def test_gradient_engine_requires_existing_warmup_models_by_default(tmp_path) -> None:
    config = ChronosTinyConfig(warmup_model_root=str(tmp_path / "missing"), seed=42)
    with pytest.raises(ModelNotFoundError, match="requires usable warmup models"):
        ChronosTinyGradientEngine(config, device="cpu")
    assert not (tmp_path / "missing").exists()


def test_warmup_persists_candidates_and_restores_base_model(tmp_path) -> None:
    engine = _engine()
    values = np.asarray([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=np.float32)
    before = {name: value.detach().clone() for name, value in engine.model.state_dict().items()}
    manifest = WarmupCandidateRunner(engine, tmp_path).run(
        values,
        learning_rates=(0.01, 0.02),
        steps=2,
        batch_size=2,
        run_id="toy",
    )
    assert manifest["schema_version"] == "ts3.gradient-warmup.v1"
    assert len(manifest["candidates"]) == 2
    assert (tmp_path / "toy" / "manifest.json").is_file()
    for candidate in manifest["candidates"]:
        assert (tmp_path / "toy" / candidate["model_dir"] / "model_state.pt").is_file()
    for name, value in engine.model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    with pytest.raises(FileExistsError):
        WarmupCandidateRunner(engine, tmp_path).run(values, steps=1, batch_size=1, run_id="toy")
