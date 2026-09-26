"""External-baseline provenance and static-deployment contracts."""

from __future__ import annotations

import pytest

from baselines.contracts import (
    BASELINE_CONTRACTS,
    baseline_contract,
    verify_static_identities,
)


def test_all_four_external_baselines_have_explicit_fidelity() -> None:
    assert set(BASELINE_CONTRACTS) == {"less", "tsds", "lp_mirror", "ltsv"}
    for contract in BASELINE_CONTRACTS.values():
        assert contract.fidelity
        assert contract.preserved_core
        assert contract.static_framework_adaptations
        if contract.source_url is None:
            assert contract.source_revision is None
            assert contract.unresolved_gaps


def test_less_contract_does_not_claim_original_parity() -> None:
    contract = baseline_contract("less")
    assert contract.fidelity == "core-signal-only"
    assert any("Adam" in gap and "intentionally omitted" in gap for gap in contract.unresolved_gaps)


def test_ltsv_contract_records_broken_official_head() -> None:
    contract = baseline_contract("ltsv")
    assert contract.source_revision == "38221dc7505ccf5ae22615f2fcba40a39b59830b"
    assert any("broken gitlinks" in gap for gap in contract.unresolved_gaps)


def test_static_identity_verifier_uses_authorized_locator() -> None:
    allowed = {(7, 1): 0, (7, 7): 1}
    verify_static_identities([(7, 1), (7, 7)], lambda fid, start: allowed[(fid, start)])
    with pytest.raises(KeyError):
        verify_static_identities([(8, 1)], lambda fid, start: allowed[(fid, start)])
