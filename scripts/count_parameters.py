#!/usr/bin/env python3
"""Print DistriSurg parameter counts for a configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distrisurg.config import load_config
from distrisurg.models import DistriSurg
from distrisurg.utils.parameters import distrisurg_parameter_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/distrisurg_dataset89.yaml"))
    parser.add_argument("--ablation-config", action="append", default=[])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = load_config(args.config, args.set, args.ablation_config)
    report = distrisurg_parameter_report(DistriSurg(config))
    for name, count in report.items():
        print(f"{name}: {count:,}")


if __name__ == "__main__":
    main()
