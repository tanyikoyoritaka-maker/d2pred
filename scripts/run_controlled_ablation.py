#!/usr/bin/env python3
"""Retired v3 runner. It remains as a fail-closed compatibility entry point."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/controlled_ablation_protocol_expert_exact_v3.json"


def main():
    state = json.loads(PROTOCOL.read_text())
    raise SystemExit(
        "Refusing to execute %s (status=%s). Historical implementations require user review; "
        "see ablation_recovery/reports/historical_ablation_recovery_report.md."
        % (state.get("protocol_id"), state.get("status"))
    )


if __name__ == "__main__":
    main()
