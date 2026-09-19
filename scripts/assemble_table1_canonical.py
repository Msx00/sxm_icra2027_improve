#!/usr/bin/env python3
"""Assemble the canonical 13-method Table-I result from audited metric files.

Pixel metrics and standardized LPIPS metrics may originate from different
evaluators.  This script merges them by normalized method/metric names,
requires N=1,392 and all six metrics for every expected method, and rejects
conflicting duplicate observations.  It writes a machine-readable summary,
held-out rankings, and two-decimal LaTeX rows.  The in-domain Hunyuan-DiT
diagnostic is retained in the table but excluded from held-out ranking/bolding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPECTED_COUNT = 1392
EXPECTED_SCENE_COUNT = 7
DIAGNOSTIC_METHOD = "Hunyuan-DiT (in-domain adapted)"
PRIMARY_METHOD = "DistriNVS (soft fusion; primary)"

GAUSSIAN_METHODS = (
    "Deform3DGS",
    "Endo-4DGS",
    "Free-SurGS",
    "SurgicalGS",
    "StereoSurGS (FS)",
    "PR-ENDO",
    "EndoGS",
    "EndoGaussian",
)
COMPLETION_METHODS = (
    "LDM",
    "LaMa",
    "LCM-LoRA",
    DIAGNOSTIC_METHOD,
)
EXPECTED_METHODS = GAUSSIAN_METHODS + COMPLETION_METHODS + (PRIMARY_METHOD,)

METRICS = (
    "psnr",
    "ssim",
    "lpips",
    "hole_psnr",
    "hole_ssim",
    "hole_lpips",
)
PIXEL_METRICS = ("psnr", "ssim", "hole_psnr", "hole_ssim")
LPIPS_METRICS = ("lpips", "hole_lpips")
LOWER_IS_BETTER = {"lpips", "hole_lpips"}

METHOD_LABELS = {
    "Deform3DGS": r"Deform3DGS~\cite{yang2024deform3dgs}",
    "Endo-4DGS": r"Endo-4DGS~\cite{huang2024endo4dgs}",
    "Free-SurGS": r"Free-SurGS~\cite{guo2024freesurgs}",
    "SurgicalGS": r"SurgicalGS~\cite{chen2025surgicalgs}",
    "StereoSurGS (FS)": r"StereoSurGS (FS)",
    "PR-ENDO": r"PR-ENDO~\cite{kaleta2025prendo}",
    "EndoGS": r"EndoGS~\cite{zhu2024endogs}",
    "EndoGaussian": r"EndoGaussian~\cite{liu2025endogaussian}",
    "LDM": r"Latent Diffusion Model (LDM)~\cite{rombach2022ldm}",
    "LaMa": r"LaMa~\cite{suvorov2022lama}",
    "LCM-LoRA": r"LCM-LoRA (endoscopy-adapted)~\cite{luo2023lcmlora}",
    DIAGNOSTIC_METHOD: (
        r"Hunyuan-DiT (in-domain adapted)$^\dagger$~\cite{li2024hunyuandit}"
    ),
    PRIMARY_METHOD: r"DistriNVS (soft fusion; primary)",
}


def compact_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


METHOD_ALIASES = {
    compact_name(name): name for name in EXPECTED_METHODS
}
METHOD_ALIASES.update(
    {
        "deform3dgaussians": "Deform3DGS",
        "endo4dgaussians": "Endo-4DGS",
        "freesurgs": "Free-SurGS",
        "freesurg": "Free-SurGS",
        "stereosurgs": "StereoSurGS (FS)",
        "stereosurgsfoundationstereo": "StereoSurGS (FS)",
        "stereosurgsfoundationstereodepth": "StereoSurGS (FS)",
        "stereosurgsfsdepth": "StereoSurGS (FS)",
        "prendo": "PR-ENDO",
        "latentdiffusionmodel": "LDM",
        "latentdiffusionmodelldm": "LDM",
        "sd15": "LDM",
        "sd15ldm": "LDM",
        "stablediffusion15": "LDM",
        "lcmloraendoscopyadapted": "LCM-LoRA",
        "lcmloraendoadapted": "LCM-LoRA",
        "hunyuan": DIAGNOSTIC_METHOD,
        "hunyuandit": DIAGNOSTIC_METHOD,
        "hunyuanditindomain": DIAGNOSTIC_METHOD,
        "distrinvs": PRIMARY_METHOD,
        "distrinvssoftfusion": PRIMARY_METHOD,
        "distrinvsoftfusion": PRIMARY_METHOD,
        "distrisurg": PRIMARY_METHOD,
    }
)

METRIC_ALIASES = {
    "psnr": "psnr",
    "ssim": "ssim",
    "lpips": "lpips",
    "holepsnr": "hole_psnr",
    "hpsnr": "hole_psnr",
    "rawholepsnr": "hole_psnr",
    "holessim": "hole_ssim",
    "hssim": "hole_ssim",
    "rawholessim": "hole_ssim",
    "holelpips": "hole_lpips",
    "hlpips": "hole_lpips",
    "rawholelpips": "hole_lpips",
}


@dataclass
class MetricValue:
    mean: float
    sample_sd: float
    sources: list[str] = field(default_factory=list)


@dataclass
class MethodValue:
    count: int | None = None
    count_sources: list[str] = field(default_factory=list)
    metrics: dict[str, MetricValue] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian-pixel-summary", type=Path, required=True)
    parser.add_argument(
        "--gaussian-lpips-summary",
        type=Path,
        nargs="+",
        required=True,
        help="One or more standardized Gaussian LPIPS summary JSONs.",
    )
    parser.add_argument("--reconstruction-summary", type=Path, required=True)
    parser.add_argument("--completion-summary", type=Path, required=True)
    parser.add_argument("--hunyuan-summary", type=Path, required=True)
    parser.add_argument("--primary-frame-metrics", type=Path, required=True)
    parser.add_argument("--primary-lpips", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conflict-atol",
        type=float,
        default=1.0e-7,
        help="Absolute tolerance for agreeing duplicate metrics (default: 1e-7).",
    )
    parser.add_argument(
        "--conflict-rtol",
        type=float,
        default=1.0e-7,
        help="Relative tolerance for agreeing duplicate metrics (default: 1e-7).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print a concise status without writing outputs.",
    )
    return parser.parse_args()


def canonical_method(value: str) -> str:
    token = compact_name(str(value).strip())
    try:
        return METHOD_ALIASES[token]
    except KeyError as error:
        raise ValueError(f"unexpected or unknown Table-I method name: {value!r}") from error


def canonical_metric(value: str) -> str | None:
    return METRIC_ALIASES.get(compact_name(str(value)))


def parse_count(value: Any, context: str) -> int:
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: invalid count {value!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{context}: invalid count {value!r}")
    return int(numeric)


def parse_finite(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected a finite number, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number, got {value!r}")
    return result


def parse_metric_value(value: Any, context: str) -> tuple[float, float]:
    if isinstance(value, dict):
        if "mean" not in value:
            raise ValueError(f"{context}: metric object is missing 'mean'")
        sd_keys = ("sample_sd", "sample_std", "std", "sd")
        present = [key for key in sd_keys if key in value]
        if not present:
            raise ValueError(
                f"{context}: metric object requires sample_sd (or sample_std/std/sd)"
            )
        mean = parse_finite(value["mean"], f"{context}.mean")
        sample_sd = parse_finite(value[present[0]], f"{context}.{present[0]}")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        mean = parse_finite(value[0], f"{context}[0]")
        sample_sd = parse_finite(value[1], f"{context}[1]")
    elif isinstance(value, str):
        match = re.fullmatch(
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            r"\s*(?:±|\+/-|\\pm)\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*",
            value,
        )
        if not match:
            raise ValueError(f"{context}: cannot parse mean±sample_sd from {value!r}")
        mean = parse_finite(match.group(1), f"{context}.mean")
        sample_sd = parse_finite(match.group(2), f"{context}.sample_sd")
    else:
        raise ValueError(
            f"{context}: expected {{mean, sample_sd}}, a two-item pair, or mean±SD"
        )
    if sample_sd < 0:
        raise ValueError(f"{context}: sample SD cannot be negative: {sample_sd}")
    return mean, sample_sd


def load_json(path: Path) -> tuple[Path, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        return resolved, json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {resolved}: {error}") from error


def extract_method_nodes(payload: Any, path: Path) -> tuple[list[tuple[str, dict]], int | None]:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: summary root must be a JSON object")
    protocol_count: int | None = None
    protocol = payload.get("protocol")
    if isinstance(protocol, dict) and protocol.get("count_per_method") is not None:
        protocol_count = parse_count(
            protocol["count_per_method"], f"{path}:protocol.count_per_method"
        )

    container: Any
    if "methods" in payload:
        container = payload["methods"]
    elif "results" in payload:
        container = payload["results"]
    elif "method" in payload and ("metrics" in payload or "summary" in payload):
        container = [payload]
    else:
        candidate = {
            key: value
            for key, value in payload.items()
            if isinstance(value, dict) and compact_name(key) in METHOD_ALIASES
        }
        if not candidate:
            raise ValueError(
                f"{path}: expected a 'methods'/'results' container or a method record"
            )
        container = candidate

    nodes: list[tuple[str, dict]] = []
    if isinstance(container, dict):
        for raw_name, node in container.items():
            if not isinstance(node, dict):
                raise ValueError(f"{path}: method {raw_name!r} must map to an object")
            nodes.append((str(raw_name), node))
    elif isinstance(container, list):
        for index, node in enumerate(container):
            if not isinstance(node, dict) or not node.get("method"):
                raise ValueError(f"{path}: method record {index} lacks a method name")
            nodes.append((str(node["method"]), node))
    else:
        raise ValueError(f"{path}: methods/results must be an object or list")
    if not nodes:
        raise ValueError(f"{path}: no method records found")
    return nodes, protocol_count


def metric_mapping(node: dict, path: Path, raw_method: str) -> dict:
    value: Any = node.get("metrics", node.get("summary", node))
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{raw_method}: metrics must be an object")
    return value


class ResultStore:
    def __init__(self, *, atol: float, rtol: float) -> None:
        if atol < 0 or rtol < 0:
            raise ValueError("conflict tolerances must be non-negative")
        self.atol = atol
        self.rtol = rtol
        self.methods: dict[str, MethodValue] = {}

    def method(self, method: str) -> MethodValue:
        return self.methods.setdefault(method, MethodValue())

    def add_count(self, method: str, count: int, source: str) -> None:
        record = self.method(method)
        if record.count is not None and record.count != count:
            raise ValueError(
                f"conflicting count for {method}: {record.count} from "
                f"{record.count_sources} versus {count} from {source}"
            )
        record.count = count
        if source not in record.count_sources:
            record.count_sources.append(source)

    def add_metric(
        self, method: str, metric: str, mean: float, sample_sd: float, source: str
    ) -> None:
        record = self.method(method)
        existing = record.metrics.get(metric)
        if existing is not None:
            agreeing = math.isclose(
                existing.mean,
                mean,
                rel_tol=self.rtol,
                abs_tol=self.atol,
            ) and math.isclose(
                existing.sample_sd,
                sample_sd,
                rel_tol=self.rtol,
                abs_tol=self.atol,
            )
            if not agreeing:
                raise ValueError(
                    f"conflicting {metric} for {method}: "
                    f"{existing.mean}±{existing.sample_sd} from {existing.sources} "
                    f"versus {mean}±{sample_sd} from {source}"
                )
            if source not in existing.sources:
                existing.sources.append(source)
            return
        record.metrics[metric] = MetricValue(mean, sample_sd, [source])

    def add_summary(self, path: Path, allowed_metrics: Iterable[str]) -> None:
        resolved, payload = load_json(path)
        nodes, protocol_count = extract_method_nodes(payload, resolved)
        allowed = set(allowed_metrics)
        for raw_method, node in nodes:
            method = canonical_method(str(node.get("method", raw_method)))
            source = str(resolved)
            count_value = node.get("count", node.get("n", protocol_count))
            if count_value is not None:
                self.add_count(
                    method,
                    parse_count(count_value, f"{resolved}:{raw_method}.count"),
                    source,
                )
            metrics = metric_mapping(node, resolved, raw_method)
            found: set[str] = set()
            for raw_metric, value in metrics.items():
                metric = canonical_metric(str(raw_metric))
                if metric is None or metric not in allowed:
                    continue
                mean, sample_sd = parse_metric_value(
                    value, f"{resolved}:{raw_method}.{raw_metric}"
                )
                if metric in found:
                    # Treat aliases within one source exactly like duplicates across sources.
                    self.add_metric(method, metric, mean, sample_sd, source)
                else:
                    self.add_metric(method, metric, mean, sample_sd, source)
                    found.add(metric)
            if not found:
                raise ValueError(
                    f"{resolved}:{raw_method}: none of the requested metrics "
                    f"{sorted(allowed)} were found"
                )

    def validate_complete(self) -> None:
        actual = set(self.methods)
        expected = set(EXPECTED_METHODS)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"expected exactly {len(expected)} Table-I methods; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for method in EXPECTED_METHODS:
            record = self.methods[method]
            if record.count != EXPECTED_COUNT:
                raise ValueError(
                    f"{method}: expected count {EXPECTED_COUNT}, got {record.count}"
                )
            missing_metrics = set(METRICS) - set(record.metrics)
            unexpected_metrics = set(record.metrics) - set(METRICS)
            if missing_metrics or unexpected_metrics:
                raise ValueError(
                    f"{method}: missing metrics={sorted(missing_metrics)}, "
                    f"unexpected metrics={sorted(unexpected_metrics)}"
                )


def add_primary_pixels(store: ResultStore, path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    values = {metric: [] for metric in PIXEL_METRICS}
    sample_ids: set[str] = set()
    scenes: set[str] = set()
    with resolved.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", *PIXEL_METRICS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{resolved}: missing columns {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            sample_id = row.get("sample_id", "").strip().strip("/")
            parts = sample_id.split("/")
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"{resolved}:{row_number}: sample_id must be 'scene/frame'"
                )
            if sample_id in sample_ids:
                raise ValueError(f"{resolved}:{row_number}: duplicate {sample_id}")
            if row.get("method", "").strip():
                method = canonical_method(row["method"])
                if method != PRIMARY_METHOD:
                    raise ValueError(
                        f"{resolved}:{row_number}: primary CSV contains {method}"
                    )
            sample_ids.add(sample_id)
            scenes.add(parts[0])
            for metric in PIXEL_METRICS:
                values[metric].append(
                    parse_finite(row[metric], f"{resolved}:{row_number}.{metric}")
                )
    if len(sample_ids) != EXPECTED_COUNT or len(scenes) != EXPECTED_SCENE_COUNT:
        raise ValueError(
            f"{resolved}: expected {EXPECTED_COUNT} unique samples in "
            f"{EXPECTED_SCENE_COUNT} scenes, got {len(sample_ids)} in {len(scenes)}"
        )
    store.add_count(PRIMARY_METHOD, len(sample_ids), str(resolved))
    for metric, metric_values in values.items():
        array = np.asarray(metric_values, dtype=np.float64)
        store.add_metric(
            PRIMARY_METHOD,
            metric,
            float(array.mean()),
            float(array.std(ddof=1)),
            str(resolved),
        )


def add_primary_lpips(store: ResultStore, path: Path) -> None:
    resolved, payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved}: primary LPIPS root must be an object")
    # Native evaluate_primary_lpips.py output has top-level count/lpips/hole_lpips.
    if all(key in payload for key in ("count", "lpips", "hole_lpips")):
        store.add_count(
            PRIMARY_METHOD,
            parse_count(payload["count"], f"{resolved}:count"),
            str(resolved),
        )
        for metric in LPIPS_METRICS:
            mean, sample_sd = parse_metric_value(
                payload[metric], f"{resolved}:{metric}"
            )
            store.add_metric(
                PRIMARY_METHOD, metric, mean, sample_sd, str(resolved)
            )
        return
    before = set(store.methods.get(PRIMARY_METHOD, MethodValue()).metrics)
    store.add_summary(resolved, LPIPS_METRICS)
    after = set(store.methods.get(PRIMARY_METHOD, MethodValue()).metrics)
    if not set(LPIPS_METRICS).issubset(after) or after == before:
        raise ValueError(
            f"{resolved}: generic primary LPIPS summary did not provide both LPIPS metrics"
        )


def serializable_summary(store: ResultStore, input_paths: dict[str, Any]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in EXPECTED_METHODS:
        record = store.methods[method]
        methods[method] = {
            "count": record.count,
            "held_out": method != DIAGNOSTIC_METHOD,
            "metrics": {
                metric: {
                    "mean": record.metrics[metric].mean,
                    "sample_sd": record.metrics[metric].sample_sd,
                    "sources": record.metrics[metric].sources,
                }
                for metric in METRICS
            },
            "count_sources": record.count_sources,
        }
    return {
        "protocol": {
            "name": "Canonical Table I metric assembly",
            "expected_methods": list(EXPECTED_METHODS),
            "count_per_method": EXPECTED_COUNT,
            "aggregation": "frame-wise mean and sample standard deviation (ddof=1)",
            "diagnostic_excluded_from_held_out_ranking": DIAGNOSTIC_METHOD,
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "duplicate_policy": "agree within configured absolute/relative tolerance",
            "inputs": input_paths,
        },
        "methods": methods,
    }


def build_rankings(store: ResultStore) -> dict[str, Any]:
    result: dict[str, Any] = {
        "excluded_from_held_out_ranking": [DIAGNOSTIC_METHOD],
        "metrics": {},
    }
    for metric in METRICS:
        reverse = metric not in LOWER_IS_BETTER
        all_order = sorted(
            EXPECTED_METHODS,
            key=lambda method: store.methods[method].metrics[metric].mean,
            reverse=reverse,
        )
        held_out_order = [
            method for method in all_order if method != DIAGNOSTIC_METHOD
        ]

        def entry(method: str, rank: int | None) -> dict[str, Any]:
            value = store.methods[method].metrics[metric]
            return {
                "method": method,
                "rank": rank,
                "mean": value.mean,
                "sample_sd": value.sample_sd,
                "held_out": method != DIAGNOSTIC_METHOD,
            }

        result["metrics"][metric] = {
            "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
            "best_held_out": held_out_order[0],
            "held_out_ranking": [
                entry(method, rank)
                for rank, method in enumerate(held_out_order, start=1)
            ],
            "all_methods_order": [
                entry(
                    method,
                    None
                    if method == DIAGNOSTIC_METHOD
                    else held_out_order.index(method) + 1,
                )
                for method in all_order
            ],
        }
    return result


def formatted_metric(
    method: str, metric: str, value: MetricValue, best_held_out: str
) -> str:
    body = f"{value.mean:.2f}\\pm{value.sample_sd:.2f}"
    if method == best_held_out:
        body = rf"\mathbf{{{body}}}"
    return f"${body}$"


def build_latex_rows(store: ResultStore, rankings: dict[str, Any]) -> str:
    best = {
        metric: rankings["metrics"][metric]["best_held_out"] for metric in METRICS
    }

    def row(method: str) -> str:
        record = store.methods[method]
        values = [
            formatted_metric(method, metric, record.metrics[metric], best[metric])
            for metric in METRICS
        ]
        return (
            f"{METHOD_LABELS[method]} & {record.count:,} & "
            + " & ".join(values)
            + r" \\" 
        )

    lines = [
        r"% Generated by assemble_table1_canonical.py; do not edit values manually.",
        r"\multicolumn{8}{l}{\textit{Gaussian reconstruction---complete coverage}} \\",
        *[row(method) for method in GAUSSIAN_METHODS],
        r"\addlinespace",
        r"\multicolumn{8}{l}{\textit{Image completion}} \\",
        *[row(method) for method in COMPLETION_METHODS],
        r"\midrule",
        row(PRIMARY_METHOD),
    ]
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, contents: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    store = ResultStore(atol=args.conflict_atol, rtol=args.conflict_rtol)

    store.add_summary(args.gaussian_pixel_summary, PIXEL_METRICS)
    for path in args.gaussian_lpips_summary:
        store.add_summary(path, LPIPS_METRICS)
    store.add_summary(args.reconstruction_summary, METRICS)
    store.add_summary(args.completion_summary, METRICS)
    store.add_summary(args.hunyuan_summary, METRICS)
    add_primary_pixels(store, args.primary_frame_metrics)
    add_primary_lpips(store, args.primary_lpips)
    store.validate_complete()

    input_paths = {
        "gaussian_pixel_summary": str(args.gaussian_pixel_summary.expanduser().resolve()),
        "gaussian_lpips_summaries": [
            str(path.expanduser().resolve()) for path in args.gaussian_lpips_summary
        ],
        "reconstruction_summary": str(args.reconstruction_summary.expanduser().resolve()),
        "completion_summary": str(args.completion_summary.expanduser().resolve()),
        "hunyuan_summary": str(args.hunyuan_summary.expanduser().resolve()),
        "primary_frame_metrics": str(args.primary_frame_metrics.expanduser().resolve()),
        "primary_lpips": str(args.primary_lpips.expanduser().resolve()),
    }
    summary = serializable_summary(store, input_paths)
    rankings = build_rankings(store)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "method_count": len(store.methods),
                    "count_per_method": EXPECTED_COUNT,
                    "best_held_out": {
                        metric: rankings["metrics"][metric]["best_held_out"]
                        for metric in METRICS
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(
        output_dir / "ranking.json",
        json.dumps(rankings, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(output_dir / "table_rows.tex", build_latex_rows(store, rankings))
    print(
        json.dumps(
            {
                "status": "written",
                "output_dir": str(output_dir),
                "method_count": len(store.methods),
                "count_per_method": EXPECTED_COUNT,
                "best_held_out": {
                    metric: rankings["metrics"][metric]["best_held_out"]
                    for metric in METRICS
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
