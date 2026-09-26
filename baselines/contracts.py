"""Unified provenance registry; method-specific contracts live in each package."""
from __future__ import annotations
from types import MappingProxyType
from typing import Callable, Iterable, Mapping
from ._contracts import BaselineContract
from .less.contract import CONTRACT as LESS_CONTRACT
from .tsds.contract import CONTRACT as TSDS_CONTRACT
from .lp_mirror.contract import CONTRACT as LP_MIRROR_CONTRACT
from .ltsv.contract import CONTRACT as LTSV_CONTRACT

BASELINE_CONTRACTS: Mapping[str, BaselineContract] = MappingProxyType({
    "less": LESS_CONTRACT,
    "tsds": TSDS_CONTRACT,
    "lp_mirror": LP_MIRROR_CONTRACT,
    "ltsv": LTSV_CONTRACT,
})

def baseline_contract(name: str) -> BaselineContract:
    try:
        return BASELINE_CONTRACTS[name]
    except KeyError as exc:
        raise ValueError("unknown baseline contract: %s" % name) from exc


def verify_static_identities(
    window_ids: Iterable[tuple[int, int]],
    locate: Callable[[int, int], int],
) -> None:
    """Prove every baseline output identity belongs to the authorized pool."""
    for fragment_id, start in window_ids:
        locate(int(fragment_id), int(start))
