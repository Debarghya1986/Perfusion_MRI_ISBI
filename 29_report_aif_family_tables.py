#!/usr/bin/env python3
"""Print four-AIF overall and family-wise reconstruction tables to the terminal.

This is a reporting-only replacement for 29_report_aif_family_tables.py.
NO LaTeX, PDF, CSV, Markdown or JSON files are generated. Nothing is overwritten.

Default methods:
  RED-CCD (nonnegative), RED-TV (nonnegative), SPINNED-style (frozen).

Inputs: per_case.csv already produced by 24_frozen_spinned_aif.py and
25_projected_red_aif.py. Optional --include-original reads saved unconstrained
RED-CCD/TV per-case metrics for matched comparisons.

Default requires exactly 120 cases (60 per family) per AIF/method.
Run from ~/ISBI_2027_Final/ or supply --repo /path/to/repository.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

CONDITIONS = ("baseline", "delayed", "broad", "sharp")
FAMILIES = ("biexponential", "pharmacokinetic")
METRICS = ("cbf_absrel_percent", "cbv_absrel_percent", "mtt_absrel_percent", "h_nrmse")
METHODS = (
    ("red_ccd_nonnegative", "RED-CCD (nonneg)"),
    ("red_tv_nonnegative", "RED-TV (nonneg)"),
    ("spinned_style_frozen", "SPINNED-style (frozen)"),
)
WITH_ORIGINAL = (
    ("red_ccd_original", "RED-CCD (original)"),
    ("red_ccd_nonnegative", "RED-CCD (nonneg)"),
    ("red_tv_original", "RED-TV (original)"),
    ("red_tv_nonnegative", "RED-TV (nonneg)"),
    ("spinned_style_frozen", "SPINNED-style (frozen)"),
)


def source_path(repo: Path, condition: str, method: str) -> Path:
    ext = repo / "outputs/v3/aif_extension_tests"
    if method == "spinned_style_frozen":
        return ext / "spinned_frozen" / condition / "full" / "per_case.csv"
    prior = method.split("_")[1]
    if method.endswith("_nonnegative"):
        return ext / "nonnegative_red" / condition / prior / "full" / "per_case.csv"
    if condition == "baseline":
        return repo / "outputs/v3/red_priors_test" / f"{prior}_init_c" / "per_case.csv"
    return (repo / "outputs/v3/aif_shift/runs" / condition / "red_priors_test"
            / f"{prior}_init_c" / "per_case.csv")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_rows(repo: Path, condition: str, method: str, expected: int):
    file = source_path(repo, condition, method)
    if file.is_file():
        rows = load_csv(file)
    elif condition != "baseline" and method.endswith("_original"):
        combined = (repo / "outputs/v3/aif_shift/runs" / condition
                    / "all_methods_test/per_case.csv")
        if not combined.is_file():
            raise FileNotFoundError(f"{file} (also checked {combined})")
        rows = [r for r in load_csv(combined)
                if r.get("method") == method.removesuffix("_original")]
        file = combined
    else:
        raise FileNotFoundError(file)

    if len(rows) != expected:
        raise ValueError(f"{condition}/{method}: found {len(rows)} cases, expected {expected}; source={file}")
    names = [r.get("case_file", "") for r in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{file}: missing or repeated case_file identifiers")
    count = Counter(r.get("family", "").strip().lower() for r in rows)
    if expected % 2 or dict(count) != {f: expected // 2 for f in FAMILIES}:
        raise ValueError(f"{file}: family counts {dict(count)}, expected {expected // 2} each")
    for r in rows:
        r["family"] = r["family"].strip().lower()
        for key in METRICS:
            try:
                val = float(r[key])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{file}: bad metric {key} for {r['case_file']}") from exc
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{file}: invalid {key}={val} for {r['case_file']}")
    return rows, file


def mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least two test cases to compute sample SD")
    avg = math.fsum(xs) / n
    sd = math.sqrt(math.fsum((x - avg) ** 2 for x in xs) / (n - 1))
    return avg, sd


def aggregate(rows: list[dict], family: str):
    selected = rows if family == "overall" else [r for r in rows if r["family"] == family]
    scores = {}
    for key in METRICS:
        scores[key] = mean_sd([float(r[key]) for r in selected])
    # Like the existing V3 table: mean of the three *case-wise* map AbsRel scores.
    objective = [sum(float(r[k]) for k in METRICS[:3]) / 3.0 for r in selected]
    return len(selected), scores, mean_sd(objective)


def print_section(title: str, loaded: dict, family: str):
    if not loaded:
        return
    print("\n" + "=" * 138)
    count = len(next(iter(loaded.values()))[0]) if family == "overall" else len(next(iter(loaded.values()))[0]) // 2
    print(f"{title}  (n={count})")
    print("=" * 138)
    print(f"{'Method':<26} {'CBF AbsRel (%)':>23} {'CBV AbsRel (%)':>23} "
          f"{'MTT AbsRel (%)':>23} {'h-NRMSE':>21} {'Mean Obj (%)':>14}")
    print("-" * 138)
    for method, (rows, label) in loaded.items():
        n, metrics, obj = aggregate(rows, family)
        cells = []
        for key in METRICS:
            m, sd = metrics[key]
            decimals = 5 if key == "h_nrmse" else 4
            cells.append(f"{m:.{decimals}f} ± {sd:.{decimals}f}")
        print(f"{label:<26} {cells[0]:>23} {cells[1]:>23} {cells[2]:>23} "
              f"{cells[3]:>21} {obj[0]:>14.4f}")
    print("-" * 138)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=".", help="Repository root (default: current directory)")
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--include-original", action="store_true", help="Also show original unconstrained RED-CCD and RED-TV")
    p.add_argument("--allow-missing", action="store_true", help="Print available complete results and identify missing files")
    p.add_argument("--expected-cases", type=int, default=120, help="Expected test cases per method and AIF")
    args = p.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    if args.expected_cases <= 2 or args.expected_cases % 2:
        p.error("--expected-cases must be even and greater than 2")
    methods = WITH_ORIGINAL if args.include_original else METHODS
    loaded_by_cond = {}
    missing = []
    across_reference = None

    # Load and validate BEFORE printing, so strict mode never produces partial tables.
    for condition in dict.fromkeys(args.conditions):
        loaded = {}
        for method, label in methods:
            try:
                rows, file = get_rows(repo, condition, method, args.expected_cases)
            except FileNotFoundError as exc:
                if not args.allow_missing:
                    p.error(f"Missing {condition}/{method}: {exc}. Use --allow-missing to show completed results only.")
                missing.append(f"{condition}/{method}: {exc}")
                continue
            loaded[method] = (rows, label)
        if loaded:
            first = next(iter(loaded))
            ref = {r["case_file"]: r["family"] for r in loaded[first][0]}
            for method, (rows, _) in loaded.items():
                check = {r["case_file"]: r["family"] for r in rows}
                if check != ref:
                    raise ValueError(f"{condition}/{method}: case IDs/families differ from {first}")
            if across_reference is None:
                across_reference = ref
            elif across_reference != ref:
                raise ValueError(f"{condition}: case identities/families differ between AIF conditions")
        loaded_by_cond[condition] = loaded

    if not any(loaded_by_cond.values()):
        p.error("No complete per_case.csv results found")

    for condition, loaded in loaded_by_cond.items():
        if not loaded:
            continue
        print("\n" + "#" * 138)
        print(f"AIF CONDITION: {condition.upper()}" + (" (NOMINAL)" if condition == "baseline" else ""))
        print("#" * 138)
        print_section("OVERALL BALANCED TEST SET", loaded, "overall")
        print_section("BI-EXPONENTIAL TEST SET", loaded, "biexponential")
        print_section("PHARMACOKINETIC TEST SET", loaded, "pharmacokinetic")

    if missing:
        print("\nMISSING RESULTS (NOT PRINTED):")
        for msg in missing:
            print(" -", msg)
    print("\nAll entries: mean ± sample SD across cases (n=120 overall; n=60 per family).")
    print("CBF/CBV/MTT are AbsRel (%), h-NRMSE is unitless; lower is better.")
    print("Mean Obj (%) = average of the three per-case map AbsRel percentages (mean only).")
    print("Reporting only: no reconstruction performed and no files saved or modified.")


if __name__ == "__main__":
    main()
