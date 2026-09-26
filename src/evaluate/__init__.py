"""Evaluation CLI and validated communication boundary.

This package validates common evaluation requests, assembles the selected
benchmark Adapter, and exposes the evaluation CLI.  It is peer to ``train``
and does not own transfer cache, training sampler, or model-update business
logic; benchmark execution and model interaction remain Adapter-owned.
"""
