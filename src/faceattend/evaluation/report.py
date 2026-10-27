"""Write evaluation reports (JSON + human-readable Markdown)."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def _f(v: Any, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v * 100:.2f}%" if pct else f"{v:.4f}"
    return str(v)


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jp, mp = out / f"eval-{stamp}.json", out / f"eval-{stamp}.md"
    jp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    L = [f"# Evaluation report ({stamp})", "", f"- Split: `{report['split']}`",
         f"- Model: `{report['model']}`", f"- Hardware: `{report['hardware']}`", ""]
    L += ["## Dataset", ""] + [f"- {k}: {v}" for k, v in report["dataset"].items()] + [""]
    fta = report["failure_to_acquire"]
    L += [f"Failure to acquire (no face / low quality): {fta['count']} ({_f(fta['rate'], True)})", ""]
    if "verification" in report:
        v = report["verification"]
        L += ["## Verification (1:1, probe vs identity template)", "",
              f"- EER: {_f(v['EER'], True)} at threshold {_f(v['EER_threshold'])}",
              f"- Threshold for FAR <= {v['target_FAR']:g}: **{_f(v['threshold_at_target_FAR'])}**", "",
              "| threshold | TAR | FRR | FAR |", "|---|---|---|---|"]
        for r in v["sweep"]:
            L.append(f"| {r['threshold']:.2f} | {_f(r['TAR'], True)} | {_f(r['FRR'], True)} | "
                     f"{_f(r['FAR'], True)} |")
        L.append("")
    if "identification" in report:
        L += ["## Open-set identification", ""] + \
             [f"- {k}: {_f(val, isinstance(val, float) and k != 'threshold')}"
              for k, val in report["identification"].items()] + [""]
    if "by_condition" in report:
        L += ["## By condition", "", "| condition | known | accuracy | unknown rejection |", "|---|---|---|---|"]
        for c, m in report["by_condition"].items():
            L.append(f"| {c} | {m['known_probes']} | {_f(m['identification_accuracy'], True)} | "
                     f"{_f(m['unknown_rejection_rate'], True)} |")
        L.append("")
    if "detection" in report:
        d = report["detection"]
        L += ["## Detection", "", f"- precision {_f(d['precision'], True)}, recall {_f(d['recall'], True)} "
              f"(TP {d['TP']}, FP {d['FP']}, FN {d['FN']}, IoU>=0.5)", ""]
    L += ["## Latency (this machine)", ""]
    for k, s in report.get("latency", {}).items():
        if s:
            L.append(f"- {k}: mean {s['mean_ms']:.1f} ms, p50 {s['p50_ms']:.1f} ms, p95 {s['p95_ms']:.1f} ms (n={s['n']})")
    if report["warnings"]:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in report["warnings"]]
    L += ["", "_All numbers are measured on the dataset above only; they do not generalise to other "
          "cameras, populations or conditions without re-evaluation._"]
    mp.write_text("\n".join(L) + "\n", encoding="utf-8")
    return jp, mp
