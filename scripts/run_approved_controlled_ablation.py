#!/usr/bin/env python3
"""Retired v3 approval launcher; no legacy approval can start training."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/controlled_ablation_protocol_expert_exact_v3.json"
APPROVAL = ROOT / "manifests/controlled_ablation_execution_approval_v3.json"


def main():
    protocol = json.loads(PROTOCOL.read_text())
    approval = json.loads(APPROVAL.read_text())
    raise SystemExit(
        "Refusing legacy launch: protocol=%s approval=%s. Review the recovered historical report before any new run."
        % (protocol.get("status"), approval.get("status"))
    )


if __name__ == "__main__":
    main()
