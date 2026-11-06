"""Shared exceptions for the canonical TSL schemas."""


class TslProtocolError(ValueError):
    """Raised for an invalid TSL schema or protocol state.

    @input: ASCII diagnostic text.
    @output: ValueError-compatible protocol exception.
    @scenario: Stop invalid resource and model protocol transitions.
    """


__all__ = ["TslProtocolError"]
