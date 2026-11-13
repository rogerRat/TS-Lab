"""Native-loss gradient probes for the newly registered foundation models."""

from gradient.base import NativeProbe


class ChronosFamilyGradientEngine(NativeProbe):
    max_gradient_batch = 8

    def __init__(self, engine, *, model_key, **kwargs):
        super().__init__(engine, model_key=model_key, **kwargs)
        self.config.context_length = int(engine.config.context_length)
        self.config.prediction_length = int(engine.config.prediction_length)


    def per_window_losses(self, windows):
        return self.engine.adapter.per_window_losses(
            windows,
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
        )


class Moirai2GradientEngine(NativeProbe):
    max_gradient_batch = 8

    def __init__(self, engine, *, model_key, **kwargs):
        super().__init__(engine, model_key=model_key, **kwargs)
        self.config.context_length = int(engine.config.context_length)
        self.config.prediction_length = int(engine.config.prediction_length)

    def per_window_losses(self, windows):
        return self.engine.adapter.per_window_losses(
            windows,
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
        )


__all__ = ["ChronosFamilyGradientEngine", "Moirai2GradientEngine"]
