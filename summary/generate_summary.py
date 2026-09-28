#!/usr/bin/env python3
"""Regenerate ALL_EXPERIMENTS_SUMMARY.{json,md} from the raw result files.

Reads every experiment JSON under RESULTS_DIR with absolute paths and emits a
machine-readable and a human-readable consolidation, each row carrying the
traceable path of the file it came from.

Usage:  python3 generate_summary.py
"""

import json
import os
from pathlib import Path

WORK_DIR = "/root/autodl-tmp/auto-baseline-v6-output/work"
RESULTS_DIR = f"{WORK_DIR}/results"
REPO_DIR = f"{WORK_DIR}/repo/SAEFit-main"
# Legacy experiments and all analysis docs live in work/results; experiments
# 12-14 are written by scripts run from the repo, so they land in the repo's own
# results/.  Both are searched, and the path recorded is the one actually found.
SEARCH_DIRS = [RESULTS_DIR, f"{REPO_DIR}/results"]
OUT_DIR = "/root/autodl-tmp"
DATE = "2026-09-26"


def resolve(name):
    """Absolute path of a result file, searching every results dir."""
    for d in SEARCH_DIRS:
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
    return None


def load(name):
    """Load a result JSON, returning None if it is absent."""
    path = resolve(name)
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def p(name):
    """Traceable path for a result file; falls back to the canonical dir."""
    return resolve(name) or f"{RESULTS_DIR}/{name}"


# ---------------------------------------------------------------- experiments
# Pre-existing experiments keep their published numbers; the three new ones are
# parsed live from their result JSONs.
LEGACY = {
    "experiment_0": {
        "name": "Calibration and Baseline",
        "priority": "Foundation",
        "reviewer_concerns": "Establishes reference baseline",
        "files": {"stage1": "calibration_stage1.json",
                  "stage2": "calibration_stage2.json"},
        "samples_per_stage": 112000,
        "conditions": 1,
        "status": "superseded_by_experiment_13",
        "caveat": "Labels derived from (dt > 0), which is constant 1; metrics "
                  "were computed on a degenerate single-class task. Use "
                  "experiment_13 instead.",
    },
    "experiment_2": {
        "name": "Amplitude Failure Stratified Analysis",
        "priority": "Medium",
        "reviewer_concerns": "Reviewer 1 amplitude reconstruction; Reviewer 2 #4-5",
        "files": {"stage1": "amplitude_failure_stage1.json",
                  "stage2": "amplitude_failure_stage2.json",
                  "analysis": "experiment2_amplitude_failure_analysis.md"},
        "samples_per_stage": 560000,
        "conditions": "5 stratification dimensions",
        "status": "valid",
    },
    "experiment_3": {
        "name": "Complete Metrics Evaluation",
        "priority": "High",
        "reviewer_concerns": "Reviewer 2 #3 complete metric set",
        "files": {"stage1": "complete_metrics_stage1.json",
                  "stage2": "complete_metrics_stage2.json",
                  "analysis": "experiment3_complete_metrics_analysis.md"},
        "samples_per_stage": 112000,
        "conditions": 1,
        "status": "valid",
    },
    "experiment_6": {
        "name": "Distribution Shift Robustness",
        "priority": "High",
        "reviewer_concerns": "Reviewer 2 #2 distribution shift robustness",
        "files": {"stage1": "rmse_stage1_corrected.json",
                  "stage2": "rmse_stage2_corrected.json",
                  "analysis": "experiment6_distribution_shift_analysis.md"},
        "samples_per_stage": 560000,
        "conditions": 5,
        "status": "valid",
    },
    "experiment_8": {
        "name": "Gain Drift Robustness",
        "priority": "Medium",
        "reviewer_concerns": "Reviewer 2 #6 gain variation robustness",
        "files": {"stage1": "gain_drift_stage1.json",
                  "stage2": "gain_drift_stage2.json",
                  "analysis": "experiment8_gain_drift_analysis.md"},
        "samples_per_stage": 784000,
        "conditions": 7,
        "status": "valid",
    },
    "experiment_9": {
        "name": "Pulse Shape Variation Robustness (Parametric)",
        "priority": "Medium",
        "reviewer_concerns": "Reviewer 1 & 2 pulse shape variation robustness",
        "files": {"stage1": "pulse_shape_parametric_stage1.json",
                  "stage2": "pulse_shape_parametric_stage2.json",
                  "analysis": "experiment9_pulse_shape_parametric_analysis.md",
                  "implementation": "parametric_pulse_generator.py"},
        "samples_per_stage": 1792000,
        "conditions": 16,
        "status": "valid",
    },
    "experiment_11": {
        "name": "OER Threshold Optimization (original)",
        "priority": "High",
        "reviewer_concerns": "Reviewer 2 #1 OER threshold selection",
        "files": {"stage1": "oer_threshold_stage1.json",
                  "stage2": "oer_threshold_stage2.json",
                  "analysis": "experiment11_oer_threshold_analysis.md"},
        "samples_per_stage": 784000,
        "conditions": 7,
        "status": "superseded_by_experiment_14",
        "caveat": "Consumed prediction dumps built from the constant (dt > 0) "
                  "label and returned 0.0 for empty high-confidence sets. Use "
                  "experiment_14 instead.",
    },
}


def parse_legacy(exp_id, meta):
    """Extract headline metrics from a pre-existing experiment's result files."""
    out = {"stage1": {}, "stage2": {}}
    for stage in ("stage1", "stage2"):
        data = load(meta["files"][stage])
        if data is None:
            out[stage] = {"error": "result file not found"}
            continue
        try:
            if exp_id == "experiment_0":
                agg = data.get("aggregate", {})
                out[stage] = {
                    "aggregate_uncalibrated": agg.get("uncalibrated"),
                    "aggregate_calibrated": agg.get("calibrated"),
                    "temperature": agg.get("temperature"),
                }
            elif exp_id == "experiment_2":
                om = data["overall_metrics"]
                out[stage] = {k: om[k] for k in om if "rmse" in k}
                out[stage]["n_samples"] = data["metadata"]["n_samples"]
            elif exp_id in ("experiment_3", "experiment_6"):
                out[stage] = data
            elif exp_id == "experiment_8":
                out[stage] = {
                    "baseline": data["baseline"],
                    "drift_factors_tested": data["metadata"].get("drift_factors_tested"),
                    "n_conditions": len(data.get("results_by_drift", [])),
                }
            elif exp_id == "experiment_9":
                b = data["baseline"]
                out[stage] = {
                    "baseline_accuracy": b["classification"]["accuracy"],
                    "baseline_a1_rmse": b["amplitude"]["a1_rmse"],
                    "baseline_a2_rmse": b["amplitude"]["a2_rmse"],
                    "baseline_t2_rmse": b["timing"]["t2_rmse"],
                    "n_conditions": len([k for k in data if k != "metadata"]),
                }
            elif exp_id == "experiment_11":
                rows = data["oer_by_threshold"]
                best = min(rows, key=lambda r: r["oer"])
                out[stage] = {
                    "n_samples": data["metadata"]["n_samples"],
                    "thresholds_evaluated": data["metadata"]["thresholds_evaluated"],
                    "lowest_oer": best["oer"],
                    "lowest_oer_threshold": best["threshold"],
                    "oer_at_0.85": next(
                        (r["oer"] for r in rows if abs(r["threshold"] - 0.85) < 1e-9),
                        None),
                }
        except (KeyError, TypeError, ValueError) as e:
            out[stage] = {"error": f"{type(e).__name__}: {e}"}
    return out


def parse_exp12():
    """Multi-seed independent repeated training."""
    d = load("multiseed_summary.json")
    if d is None:
        return None
    across, deltas = d["across_seeds"], d["paired_stage2_minus_stage1"]
    entry = {
        "metadata": {
            "name": "Multi-Seed Independent Repeated Training",
            "priority": "High",
            "reviewer_concerns": "single-seed results; run-to-run variability unknown",
            "seeds": d["metadata"]["seeds_found"],
            "status": "valid",
        },
        "file_paths": {
            "summary": p("multiseed_summary.json"),
            "per_seed": p("multiseed_per_seed.json"),
            "analysis": p("experiment12_multiseed_analysis.md"),
            "train_script": f"{REPO_DIR}/run_multiseed.sh",
            "aggregate_script": f"{REPO_DIR}/aggregate_multiseed.py",
        },
        "across_seeds": {},
        "paired_stage2_minus_stage1": {},
    }
    for stage, e in across.items():
        entry["across_seeds"][stage] = {
            "n_seeds": e["n_seeds"], "seeds": e["seeds"],
            "clean": e["clean"], "shifted_combined": e["shifted_combined"],
            "regression": e["regression"], "temperature": e["temperature"],
        }
    for metric, s in deltas["clean"].items():
        if s:
            entry["paired_stage2_minus_stage1"][metric] = s
    entry["regression_delta"] = deltas.get("regression", {})
    return entry


def parse_exp13():
    """Unified temperature scaling."""
    out = {
        "metadata": {
            "name": "Unified Temperature Scaling Calibration",
            "priority": "High",
            "reviewer_concerns": "unified post-hoc calibration protocol",
            "status": "valid",
            "supersedes": "experiment_0",
        },
        "file_paths": {
            "stage1": p("temp_scaling_stage1.json"),
            "stage2": p("temp_scaling_stage2.json"),
            "analysis": p("experiment13_temperature_scaling_analysis.md"),
            "script": f"{REPO_DIR}/evaluate_temperature_scaling.py",
            "shared_core": f"{REPO_DIR}/lab/utils/unified_eval.py",
        },
        "stages": {},
    }
    for stage in ("stage1", "stage2"):
        d = load(f"temp_scaling_{stage}.json")
        if d is None:
            out["stages"][stage] = {"error": "result file not found"}
            continue
        out["stages"][stage] = {
            "temperature_fits": d["temperature_fits"],
            "clean": d["per_condition"]["clean"],
            "aggregate_over_named_shift_conditions":
                d["aggregate_over_shift_conditions"],
            "per_condition_accuracy": {
                k: v["uncalibrated"]["accuracy"]
                for k, v in d["per_condition"].items()},
            "per_condition_ece_uncal_to_cal": {
                k: [v["uncalibrated"]["ece"],
                    v["calibrated_clean_fit"]["ece"],
                    v["calibrated_shifted_fit"]["ece"]]
                for k, v in d["per_condition"].items()},
        }
    return out


def parse_exp14():
    """OER threshold sensitivity with unified definition."""
    out = {
        "metadata": {
            "name": "OER Threshold Sensitivity (Unified OER Definition)",
            "priority": "High",
            "reviewer_concerns": "OER threshold sensitivity; unify OER definition",
            "status": "valid",
            "supersedes": "experiment_11",
        },
        "file_paths": {
            "stage1": p("oer_sensitivity_stage1.json"),
            "stage2": p("oer_sensitivity_stage2.json"),
            "analysis": p("experiment14_oer_sensitivity_analysis.md"),
            "script": f"{REPO_DIR}/evaluate_oer_sensitivity.py",
            "shared_core": f"{REPO_DIR}/lab/utils/unified_eval.py",
        },
        "oer_definitions": None,
        "stages": {},
    }
    for stage in ("stage1", "stage2"):
        d = load(f"oer_sensitivity_{stage}.json")
        if d is None:
            out["stages"][stage] = {"error": "result file not found"}
            continue
        out["oer_definitions"] = d["metadata"]["oer_definitions"]
        out["stages"][stage] = {
            "temperature": d["metadata"]["temperature"],
            "thresholds": d["metadata"]["thresholds"],
            "n_bootstrap": d["metadata"]["n_bootstrap"],
            "variants": {
                k: {"summary": v["summary"],
                    "oer_by_threshold": v["oer_by_threshold"],
                    "oer_conditional_range": v["oer_conditional_range"],
                    "aurc": v["aurc"]}
                for k, v in d["variants"].items()},
        }
    return out


# ----------------------------------------------------------------- assembly
def build():
    """Assemble the full consolidation dict."""
    experiments = {}
    for exp_id, meta in LEGACY.items():
        entry = {
            "metadata": {
                "name": meta["name"],
                "priority": meta["priority"],
                "reviewer_concerns": meta["reviewer_concerns"],
                "samples_per_stage": meta["samples_per_stage"],
                "conditions": meta["conditions"],
                "status": meta["status"],
            },
            "file_paths": {k: p(v) for k, v in meta["files"].items()},
            "results": parse_legacy(exp_id, meta),
        }
        if "caveat" in meta:
            entry["metadata"]["caveat"] = meta["caveat"]
        experiments[exp_id] = entry

    for exp_id, parsed in (("experiment_12", parse_exp12()),
                           ("experiment_13", parse_exp13()),
                           ("experiment_14", parse_exp14())):
        if parsed is None:
            experiments[exp_id] = {
                "metadata": {"status": "pending",
                             "note": "training/aggregation not yet complete"}}
        else:
            experiments[exp_id] = parsed

    return {
        "metadata": {
            "title": "SAEFit Baseline Reproduction - Consolidated Experiment Results",
            "generated": DATE,
            "generator": f"{OUT_DIR}/generate_summary.py",
            "results_dir": RESULTS_DIR,
            "repo_dir": f"{REPO_DIR}",
            "dataset_dir": f"{RESULTS_DIR}/../data",
            "n_experiments": len(experiments),
            "stages": {
                "stage1": "Timing-Focused (tau=5.0, tauc=0.05, lam=0.01, "
                          "lr=3e-4, 100 epochs)",
                "stage2": "Amplitude-Focused (tau=3.0, tauc=0.05, lam=0.05, "
                          "lr=2e-4, 100 epochs)",
            },
            "environment": "PyTorch Lightning 2.5.0 / torch 2.8.0+cu128, "
                           "RTX 4090 (49 GB)",
            "unified_protocol": {
                "module": f"{REPO_DIR}/lab/utils/unified_eval.py",
                "task": "balanced binary classification, single pulse -> label 0, "
                        "double pulse -> label 1, as in control/saefit.py::test_step",
                "n_test_samples": 224000,
                "oer_conditional": "P(yhat != y | conf >= tau); NaN when no "
                                   "sample clears tau",
                "oer_absolute": "P(yhat != y AND conf >= tau); monotone "
                                "non-increasing in tau",
                "ece_bins": 15,
                "determinism": "seed_eval_dataset() pins np.random and "
                               "torch.manual_seed before every eval dataloader; "
                               "verified bit-identical across runs",
            },
        },
        "experiments": experiments,
    }


# ------------------------------------------------------------------ markdown
def fmt(v, sig=6):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "NaN"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.3e}"
        return f"{v:.{sig}f}".rstrip("0").rstrip(".")
    return str(v)


OVERVIEW = [
    ("0", "Calibration and Baseline", "112,000", "1",
     "superseded by 13", "calibration_stage{1,2}.json"),
    ("2", "Amplitude Failure Stratified", "560,000", "5 strata",
     "valid", "amplitude_failure_stage{1,2}.json"),
    ("3", "Complete Metrics", "112,000", "1",
     "valid", "complete_metrics_stage{1,2}.json"),
    ("6", "Distribution Shift Robustness", "560,000", "5",
     "valid", "rmse_stage{1,2}_corrected.json"),
    ("8", "Gain Drift Robustness", "784,000", "7",
     "valid", "gain_drift_stage{1,2}.json"),
    ("9", "Pulse Shape Variation (Parametric)", "1,792,000", "16",
     "valid", "pulse_shape_parametric_stage{1,2}.json"),
    ("11", "OER Threshold Optimization (original)", "784,000", "7",
     "superseded by 14", "oer_threshold_stage{1,2}.json"),
    ("12", "Multi-Seed Independent Repeated Training", "224,000/seed",
     "seeds x 2 stages", "NEW", "multiseed_summary.json"),
    ("13", "Unified Temperature Scaling", "224,000", "10",
     "NEW", "temp_scaling_stage{1,2}.json"),
    ("14", "OER Threshold Sensitivity (unified OER)", "224,000",
     "12 tau x 4 variants", "NEW", "oer_sensitivity_stage{1,2}.json"),
]


def md_header(d):
    m = d["metadata"]
    L = [f"# {m['title']}", "",
         f"Generated {m['generated']} by `{m['generator']}`.", "",
         f"- Results directory: `{m['results_dir']}`",
         f"- Repository: `{m['repo_dir']}`",
         f"- Datasets: `{m['dataset_dir']}`",
         f"- Environment: {m['environment']}",
         f"- Stage 1: {m['stages']['stage1']}",
         f"- Stage 2: {m['stages']['stage2']}",
         "",
         "## Overview", "",
         "| Exp | Name | Samples/stage | Conditions | Status | Result files |",
         "|---|---|---|---|---|---|"]
    for row in OVERVIEW:
        L.append("| " + " | ".join(
            [row[0], row[1], row[2], row[3], row[4], f"`{row[5]}`"]) + " |")
    u = m["unified_protocol"]
    L += ["", "## Unified evaluation protocol (experiments 12-14)", "",
          f"Shared implementation: `{u['module']}`", "",
          f"- Task: {u['task']}",
          f"- Test samples per condition: {u['n_test_samples']:,}",
          f"- `OER_cond(tau)` = {u['oer_conditional']}",
          f"- `OER_abs(tau)` = {u['oer_absolute']}",
          f"- ECE bins: {u['ece_bins']}",
          f"- Determinism: {u['determinism']}",
          "",
          "Experiments 0 and 11 predate this protocol. They derived labels from",
          "`(dt > 0)`, which is constant 1 for every sample `PulsePrepare`",
          "produces, so their accuracy/ECE/OER numbers were measured on a",
          "degenerate single-class task. They are retained for traceability but",
          "superseded by experiments 13 and 14.", ""]
    return L


def md_legacy(exp_id, e):
    m, L = e["metadata"], []
    n = exp_id.split("_")[1]
    L += [f"## Experiment {n}: {m['name']}", "",
          f"- Priority: {m['priority']}",
          f"- Reviewer concerns: {m['reviewer_concerns']}",
          f"- Samples per stage: {m['samples_per_stage']:,}",
          f"- Conditions: {m['conditions']}",
          f"- Status: **{m['status']}**"]
    if "caveat" in m:
        L.append(f"- Caveat: {m['caveat']}")
    L += ["", "Traceable paths:", ""]
    for k, v in e["file_paths"].items():
        L.append(f"- `{k}`: `{v}`")
    L.append("")
    for stage in ("stage1", "stage2"):
        r = e["results"].get(stage, {})
        if "error" in r:
            L += [f"### {stage}", "", f"_{r['error']}_", ""]
            continue
        L += [f"### {stage}", "", "| Metric | Value |", "|---|---|"]
        for k, v in flatten(r).items():
            L.append(f"| {k} | {fmt(v)} |")
        L.append("")
    return L


def md_exp13(e):
    m = e["metadata"]
    L = [f"## Experiment 13: {m['name']}", "",
         f"- Priority: {m['priority']}",
         f"- Reviewer concerns: {m['reviewer_concerns']}",
         f"- Status: **{m['status']}** (supersedes {m['supersedes']})", "",
         "Traceable paths:", ""]
    for k, v in e["file_paths"].items():
        L.append(f"- `{k}`: `{v}`")
    L += ["", "Temperature is fitted on the validation split by minimizing NLL "
          "over a log-space grid plus golden-section refinement. Two fit "
          "sources are reported: `clean` (the headline protocol) and `shifted` "
          "(fitted under the `combined` perturbation, an upper bound on what a "
          "single temperature can recover).", ""]
    for stage in ("stage1", "stage2"):
        s = e["stages"].get(stage, {})
        if "error" in s:
            L += [f"### {stage}", "", f"_{s['error']}_", ""]
            continue
        f = s["temperature_fits"]
        L += [f"### {stage}", "",
              f"T (clean fit) = {fmt(f['clean']['temperature'])}, "
              f"T (shifted fit) = {fmt(f['shifted']['temperature'])}", "",
              "| Condition | Accuracy | ECE uncal | ECE clean-fit | "
              "ECE shift-fit |", "|---|---|---|---|---|"]
        for cond, acc in s["per_condition_accuracy"].items():
            ece = s["per_condition_ece_uncal_to_cal"][cond]
            L.append(f"| {cond} | {fmt(acc)} | {fmt(ece[0])} | "
                     f"{fmt(ece[1])} | {fmt(ece[2])} |")
        L += ["", "Aggregate over the five named paper shift conditions:", "",
              "| Variant | Accuracy | ECE | NLL | E-AURC |",
              "|---|---|---|---|---|"]
        for variant, agg in s["aggregate_over_named_shift_conditions"].items():
            L.append(f"| {variant} | {fmt(agg['accuracy'])} | "
                     f"{fmt(agg['ece'])} | {fmt(agg['nll'])} | "
                     f"{fmt(agg['e_aurc'])} |")
        L.append("")
    return L


def md_exp14(e):
    m = e["metadata"]
    L = [f"## Experiment 14: {m['name']}", "",
         f"- Priority: {m['priority']}",
         f"- Reviewer concerns: {m['reviewer_concerns']}",
         f"- Status: **{m['status']}** (supersedes {m['supersedes']})", "",
         "Traceable paths:", ""]
    for k, v in e["file_paths"].items():
        L.append(f"- `{k}`: `{v}`")
    if e.get("oer_definitions"):
        L += ["", "Unified OER definition:", ""]
        for k, v in e["oer_definitions"].items():
            L.append(f"- `{k}`: {v}")
    for stage in ("stage1", "stage2"):
        s = e["stages"].get(stage, {})
        if "error" in s:
            L += ["", f"### {stage}", "", f"_{s['error']}_", ""]
            continue
        L += ["", f"### {stage} (T = {fmt(s['temperature'])}, "
              f"{s['n_bootstrap']} bootstrap resamples)", "",
              "Threshold sweep, clean test data, uncalibrated:", "",
              "| tau | OER_cond | 95% CI | OER_abs | Coverage | N selected | "
              "N errors |", "|---|---|---|---|---|---|---|"]
        for row in s["variants"]["clean/uncalibrated"]["oer_by_threshold"]:
            L.append(
                f"| {row['threshold']:.4f} | {fmt(row['oer_conditional'])} | "
                f"[{fmt(row['oer_conditional_ci95']['lower'])}, "
                f"{fmt(row['oer_conditional_ci95']['upper'])}] | "
                f"{fmt(row['oer_absolute'])} | {fmt(row['coverage'])} | "
                f"{row['n_selected']:,} | {row['n_errors_selected']:,} |")
        L += ["", "Summary across variants:", "",
              "| Variant | Accuracy | OER@0.85 | OER_cond min | max | "
              "Spread | AURC | E-AURC |", "|---|---|---|---|---|---|---|---|"]
        for k, v in s["variants"].items():
            rng, aurc, summ = v["oer_conditional_range"], v["aurc"], v["summary"]
            spread = (f"{rng['spread_factor']:.1f}x"
                      if rng.get("spread_factor") else "n/a")
            L.append(f"| {k} | {fmt(summ['accuracy'])} | "
                     f"{fmt(summ['oer_conditional@0.85'])} | "
                     f"{fmt(rng['min'])} | {fmt(rng['max'])} | {spread} | "
                     f"{fmt(aurc['aurc'])} | {fmt(aurc['e_aurc'])} |")
        L.append("")
    return L


def md_footer(d):
    L = ["## Reviewer coverage map", "",
         "| Reviewer concern | Experiment(s) |", "|---|---|",
         "| Reviewer 2 #1 OER threshold selection | 14 (11 superseded) |",
         "| Reviewer 2 #2 distribution shift robustness | 6, 13 |",
         "| Reviewer 2 #3 complete metric set | 3 |",
         "| Reviewer 2 #4-5 amplitude reconstruction | 2 |",
         "| Reviewer 2 #6 gain variation robustness | 8 |",
         "| Reviewer 1 & 2 pulse shape variation | 9 |",
         "| Unified post-hoc calibration | 13 (0 superseded) |",
         "| Single-seed results / run-to-run variability | 12 |",
         "", "## How to cite a number from this file", "",
         "Every experiment section lists the absolute path of the JSON it was",
         "parsed from. To trace a value: open that JSON and follow the dotted",
         "key shown in the table. To regenerate everything from raw results:",
         "", "```bash",
         f"python3 {d['metadata']['generator']}", "```", "",
         "To re-run an individual experiment, see the `script` path in its",
         "section; experiments 12-14 all share "
         "`lab/utils/unified_eval.py` as their evaluation core.", ""]
    return L


def emit_md(d):
    L = md_header(d)
    for exp_id in ("experiment_0", "experiment_2", "experiment_3",
                   "experiment_6", "experiment_8", "experiment_9",
                   "experiment_11"):
        L += md_legacy(exp_id, d["experiments"][exp_id])
    for exp_id, fn in (("experiment_12", md_exp12),
                       ("experiment_13", md_exp13),
                       ("experiment_14", md_exp14)):
        e = d["experiments"][exp_id]
        if e["metadata"].get("status") == "pending":
            n = exp_id.split("_")[1]
            L += [f"## Experiment {n}", "",
                  f"_{e['metadata']['note']}_", ""]
        else:
            L += fn(e)
    L += md_footer(d)
    return "\n".join(L)


def main():
    d = build()
    json_path = os.path.join(OUT_DIR, "ALL_EXPERIMENTS_SUMMARY.json")
    md_path = os.path.join(OUT_DIR, "ALL_EXPERIMENTS_SUMMARY.md")
    with open(json_path, "w") as f:
        json.dump(d, f, indent=2, allow_nan=True)
    with open(md_path, "w") as f:
        f.write(emit_md(d))

    print(f"experiments: {d['metadata']['n_experiments']}")
    for exp_id, e in d["experiments"].items():
        status = e["metadata"].get("status", "?")
        errs = [k for k, v in (e.get("results") or {}).items()
                if isinstance(v, dict) and "error" in v]
        errs += [k for k, v in (e.get("stages") or {}).items()
                 if isinstance(v, dict) and "error" in v]
        flag = f"  ERRORS: {errs}" if errs else ""
        print(f"  {exp_id:15s} {status}{flag}")
    print(f"wrote {json_path} ({os.path.getsize(json_path):,} bytes)")
    print(f"wrote {md_path} ({os.path.getsize(md_path):,} bytes)")


def flatten(obj, prefix="", depth=0, maxdepth=3):
    """Flatten a nested dict into dotted keys, skipping long lists."""
    out = {}
    if not isinstance(obj, dict) or depth > maxdepth:
        return {prefix or "value": obj}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, depth + 1, maxdepth))
        elif isinstance(v, list):
            if len(v) <= 8 and all(not isinstance(x, (dict, list)) for x in v):
                out[key] = "[" + ", ".join(fmt(x) for x in v) + "]"
            else:
                out[key] = f"<{len(v)} entries>"
        else:
            out[key] = v
    return out


def md_exp12(e):
    m = e["metadata"]
    L = [f"## Experiment 12: {m['name']}", "",
         f"- Priority: {m['priority']}",
         f"- Reviewer concerns: {m['reviewer_concerns']}",
         f"- Status: **{m['status']}**", "",
         "Traceable paths:", ""]
    for k, v in e["file_paths"].items():
        L.append(f"- `{k}`: `{v}`")
    L += ["", "### Across-seed statistics (clean test data, uncalibrated)", "",
          "| Stage | n seeds | seeds | Accuracy (mean +- std) | 95% CI | ECE | "
          "OER@0.85 | T |", "|---|---|---|---|---|---|---|---|"]
    for stage, a in e["across_seeds"].items():
        c = a["clean"]
        acc, ece, oer = c["accuracy"], c["ece"], c["oer_conditional@0.85"]
        L.append(
            f"| {stage} | {a['n_seeds']} | "
            f"{', '.join(str(s) for s in a['seeds'])} | "
            f"{fmt(acc['mean'])} +- {fmt(acc['std'])} | "
            f"[{fmt(acc['ci95_lower'])}, {fmt(acc['ci95_upper'])}] | "
            f"{fmt(ece['mean'])} +- {fmt(ece['std'])} | "
            f"{fmt(oer['mean'])} +- {fmt(oer['std'])} | "
            f"{fmt(a['temperature']['mean'])} |")
    L += ["", "### Regression metrics across seeds", "",
          "| Stage | dt RMSE (samples) | a1 RMSE (norm) | a2 RMSE (norm) |",
          "|---|---|---|---|"]
    for stage, a in e["across_seeds"].items():
        g = a["regression"]
        L.append(f"| {stage} | "
                 + " | ".join(
                     f"{fmt(g[k]['mean'])} +- {fmt(g[k]['std'])}"
                     for k in ("dt_rmse_samples", "a1_rmse_norm", "a2_rmse_norm"))
                 + " |")
    L += ["", "### Paired stage2 - stage1 deltas (same seed, clean data)", "",
          "| Metric | Mean delta | 95% CI | n | Significant at 95% |",
          "|---|---|---|---|---|"]
    for metric, s in e["paired_stage2_minus_stage1"].items():
        L.append(f"| {metric} | {s['mean']:+.6g} | "
                 f"[{s['ci95_lower']:+.6g}, {s['ci95_upper']:+.6g}] | "
                 f"{s['n']} | {'yes' if s['significant_at_95'] else 'no'} |")
    for metric, s in (e.get("regression_delta") or {}).items():
        if s:
            L.append(f"| {metric} | {s['mean']:+.6g} | "
                     f"[{s['ci95_lower']:+.6g}, {s['ci95_upper']:+.6g}] | "
                     f"{s['n']} | {'yes' if s['significant_at_95'] else 'no'} |")
    L.append("")
    return L


if __name__ == "__main__":
    main()
