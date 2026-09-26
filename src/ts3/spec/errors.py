"""Shared exceptions for the canonical TS3 schemas."""


class TS3ProtocolError(ValueError):
    """Raised for an invalid TS3 schema or protocol state.

    @input: ASCII diagnostic text.
    @output: ValueError-compatible protocol exception.
    @scenario: Stop invalid resource and model protocol transitions.
    """


__all__ = ["TS3ProtocolError"]
