#!/usr/bin/env python
"""Regenerate the UDA/TTA benchmark tables from corrected inference outputs.

Reads the audit-fixed inference groups under ``results/inference_auditfix/`` and
recomputes the per-source-block benchmark tables that appear in
``UDA_BENCHMARK_STATUS.md``.

Table columns (per the status doc):
    Method | Mean acc | Delta | Macro-F1 | In-dist | Far >=43

Definitions:
    - Mean acc / Macro-F1: mean over all evaluated weeks present in the group.
      Macro-F1 is sklearn f1_score(average='macro') computed per week then
      averaged across weeks.
    - In-dist: accuracy on the source week's own test week
      (week 01 for week_1, week 16 for week_16, week 44 for QUIC).
    - Far >=43: mean accuracy over weeks 43..52 (TLS only; QUIC has no far span).
    - Delta: this method's Mean acc minus the vanilla (source-only) Mean acc,
      computed on the set of weeks SHARED by both groups (so a partial method
      is compared fairly against vanilla on the same weeks).

Groups that are missing weeks (relative to the vanilla/reference group for that
source block) or that contain a corrupt npz are reported as
"(incomplete: X/Y weeks)" rather than a number.

Output dir layout:
    results/inference_auditfix/<tag>_<method>_bs<batch>/WEEK-2022-NN.npz
where <tag> in {week_1, week_16, WEEK-2022-44}.

This is a read-only analysis script: it loads npz files and prints markdown.
It does not write, delete, or submit anything.

Run from the repo root:
    python scripts/analysis/regen_benchmark_table.py
"""

import os
import sys

import numpy as np

try:
    from sklearn.metrics import f1_score
except Exception as exc:  # pragma: no cover - environment guard
    print(f"ERROR: could not import sklearn.metrics.f1_score: {exc}", file=sys.stderr)
    raise

# --- locate repo root + inference dir -------------------------------------------------
_THIS = os.path.abspath(__file__)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))
INFER_DIR = os.path.join(REPO_ROOT, "results", "inference_auditfix")

# --- source-block definitions ---------------------------------------------------------
# Each block: tag (dir prefix), human title, source week index, eval-week range,
# and the "far" week range (None when not applicable, e.g. QUIC).
WEEK_FMT = "WEEK-2022-{:02d}.npz"


def week_name(idx):
    return WEEK_FMT.format(idx)


BLOCKS = [
    # Forward-only evaluation: a source model is scored only on its OWN week and
    # later weeks (the drift-monitoring scenario). Evaluating a source model on
    # EARLIER weeks ("predicting the past") is backward transfer and is dropped — it
    # is not the deployment setting and it inflates means with easy pre-source weeks.
    {
        "tag": "week_1",
        "title": "CESNET-TLS-Year22 — week-1 source (forward weeks 1–52, 180 classes)",
        "source_week": 1,
        "eval_weeks": list(range(1, 53)),
        "far_weeks": list(range(43, 53)),
    },
    {
        "tag": "week_16",
        "title": "CESNET-TLS-Year22 — week-16 source (forward weeks 16–52, 180 classes)",
        "source_week": 16,
        "eval_weeks": list(range(16, 53)),
        "far_weeks": list(range(43, 53)),
    },
    {
        "tag": "WEEK-2022-44",
        "title": "CESNET-QUIC22 — week-44 source (4 weeks, secondary)",
        "source_week": 44,
        "eval_weeks": list(range(44, 48)),
        "far_weeks": None,
    },
]

# Method rows in display order. Each entry: (display label, method token, batch).
# The first row (vanilla) is the source-only reference used for Delta.
METHOD_ROWS = [
    ("Source-only", "vanilla", 64),
    ("AdaBN", "bnstats", 64),
    ("TENT (bs64)", "tent", 64),
    ("TENT (bs256)", "tent", 256),
    ("CoTTA (bs64)", "cotta", 64),
    ("CoTTA (bs256)", "cotta", 256),
]


def group_dir(tag, method, batch):
    return os.path.join(INFER_DIR, f"{tag}_{method}_bs{batch}")


def load_week(path):
    """Return (true_labels, pred_labels) or raise on corrupt/missing keys."""
    with np.load(path) as d:
        return d["true_labels"], d["pred_labels"]


def collect_group(tag, method, batch, eval_weeks):
    """Scan a group dir; return dict week_idx -> (acc, macro_f1) for weeks that
    loaded cleanly, plus a list of weeks that are missing or corrupt.
    """
    gdir = group_dir(tag, method, batch)
    per_week = {}
    bad = []  # weeks missing or corrupt
    if not os.path.isdir(gdir):
        return per_week, list(eval_weeks), gdir  # whole group absent
    for w in eval_weeks:
        path = os.path.join(gdir, week_name(w))
        if not os.path.isfile(path):
            bad.append(w)
            continue
        try:
            y_true, y_pred = load_week(path)
            if y_true.shape[0] == 0 or y_true.shape != y_pred.shape:
                bad.append(w)
                continue
            acc = float(np.mean(y_true == y_pred))
            mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            per_week[w] = (acc, mf1)
        except Exception:
            bad.append(w)
    return per_week, bad, gdir


def mean_or_none(vals):
    return float(np.mean(vals)) if vals else None


def fmt(x, plus=False):
    if x is None:
        return "—"
    if plus:
        return f"{x:+.3f}"
    return f"{x:.3f}"


def render_block(block):
    tag = block["tag"]
    eval_weeks = block["eval_weeks"]
    far_weeks = block["far_weeks"]
    src = block["source_week"]
    n_expected = len(eval_weeks)

    # Reference (vanilla) group first.
    ref_label, ref_method, ref_batch = METHOD_ROWS[0]
    ref_pw, ref_bad, ref_dir = collect_group(tag, ref_method, ref_batch, eval_weeks)
    ref_acc_by_week = {w: v[0] for w, v in ref_pw.items()}
    ref_mean = mean_or_none([v[0] for v in ref_pw.values()])

    lines = []
    lines.append(f"## {block['title']}")
    lines.append("")
    lines.append("| Method | Mean acc | Δ | Macro-F1 | In-dist | Far ≥43 |")
    lines.append("|---|--:|--:|--:|--:|--:|")

    for label, method, batch in METHOD_ROWS:
        pw, bad, gdir = collect_group(tag, method, batch, eval_weeks)
        n_have = len(pw)

        if n_have == 0:
            # Whole group absent / unreadable.
            note = f"(incomplete: 0/{n_expected} weeks)"
            lines.append(f"| {label} | {note} | — | — | — | — |")
            continue

        incomplete = n_have < n_expected

        accs = [v[0] for v in pw.values()]
        mf1s = [v[1] for v in pw.values()]
        mean_acc = mean_or_none(accs)
        macro_f1 = mean_or_none(mf1s)

        # In-dist = source week accuracy.
        in_dist = pw[src][0] if src in pw else None

        # Far = mean acc over far weeks present.
        if far_weeks is None:
            far_val = None
            far_str = "n/a"
        else:
            far_present = [pw[w][0] for w in far_weeks if w in pw]
            far_val = mean_or_none(far_present)
            far_str = fmt(far_val)
            # Flag if far span is itself partial.
            if 0 < len(far_present) < len(far_weeks):
                far_str = f"{far_str}*"

        # Delta vs vanilla on SHARED weeks.
        is_ref = method == ref_method and batch == ref_batch
        if is_ref:
            delta_str = "—"
        elif not ref_pw:
            delta_str = "(no vanilla ref)"
        else:
            shared = sorted(set(pw) & set(ref_acc_by_week))
            if not shared:
                delta_str = "(no shared weeks)"
            else:
                m_self = np.mean([pw[w][0] for w in shared])
                m_ref = np.mean([ref_acc_by_week[w] for w in shared])
                delta_str = fmt(float(m_self - m_ref), plus=True)

        if incomplete:
            mean_str = f"(incomplete: {n_have}/{n_expected} weeks)"
            # Still show the rest so partial output is informative.
            lines.append(
                f"| {label} | {mean_str} | {delta_str} | {fmt(macro_f1)} | "
                f"{fmt(in_dist)} | {far_str} |"
            )
        else:
            lines.append(
                f"| {label} | {fmt(mean_acc)} | {delta_str} | {fmt(macro_f1)} | "
                f"{fmt(in_dist)} | {far_str} |"
            )

    # Footnote about reference completeness / partial-Delta semantics.
    notes = []
    if ref_bad:
        notes.append(
            f"vanilla reference itself missing {len(ref_bad)}/{n_expected} weeks "
            f"(Δ for partial rows computed on weeks shared with vanilla)"
        )
    if far_weeks is not None:
        notes.append("* = Far span partial (fewer than 10 of weeks 43–52 present)")
    if notes:
        lines.append("")
        lines.append("_" + "; ".join(notes) + "._")

    return "\n".join(lines)


# Training-time UDA methods scored on the DIAGONAL: each target week N (≥16) is scored by
# its OWN source-16→week-N aligned model (one model per week, transductive — the most
# favorable setting), compared to the single frozen week-16 source-only model on the SAME
# forward weeks. (label, results-dir-name) — skipped automatically if the dir is absent.
DIAG_METHODS = [
    ("DANN (diagonal)", "dann_fwd_w16_diagonal"),
    ("CORAL (diagonal)", "coral_fwd_w16_diagonal"),
    ("MMD (diagonal)", "mmd_fwd_w16_diagonal"),
]


def _load_diag(dir_name, weeks):
    d = os.path.join(REPO_ROOT, "results", "inference", dir_name)
    if not os.path.isdir(d):
        return {}
    out = {}
    for w in weeks:
        p = os.path.join(d, week_name(w))
        if os.path.isfile(p):
            try:
                yt, yp = load_week(p)
                out[w] = (float(np.mean(yt == yp)),
                          float(f1_score(yt, yp, average="macro", zero_division=0)))
            except Exception:
                pass
    return out


def render_uda_diagonal():
    """Combined diagonal table for the training-time UDA methods (DANN, CORAL) vs the
    frozen week-16 source-only model, forward weeks 16–52."""
    src = 16
    fwd = list(range(src, 53))
    van_pw, _, _ = collect_group("week_16", "vanilla", 64, fwd)
    if not van_pw:
        return ""
    methods = [(lbl, _load_diag(dn, fwd)) for lbl, dn in DIAG_METHODS]
    methods = [(lbl, d) for lbl, d in methods if d]
    if not methods:
        return ""

    def mean_acc(d, ws):
        return float(np.mean([d[w][0] for w in ws])) if ws else None

    def mean_f1(d, ws):
        return float(np.mean([d[w][1] for w in ws])) if ws else None

    lines = ["## UDA training-time alignment (diagonal) — week-16 source",
             "",
             "_Each week scored by its own source-16→week-N aligned model (transductive, one "
             "model per week — the most favorable setting for these methods); forward weeks "
             "16–52. Δ vs the frozen week-16 source-only model on shared weeks._",
             "",
             "| Method | Mean acc | Δ | Macro-F1 | Far ≥43 |",
             "|---|--:|--:|--:|--:|"]
    # Source-only reference over the full forward range.
    all_shared = sorted(set(van_pw))
    far_v = [w for w in all_shared if w >= 43]
    lines.append(f"| Source-only (week-16) | {fmt(mean_acc(van_pw, all_shared))} | — | "
                 f"{fmt(mean_f1(van_pw, all_shared))} | {fmt(mean_acc(van_pw, far_v))} |")
    for lbl, diag in methods:
        shared = sorted(set(diag) & set(van_pw))
        far = [w for w in shared if w >= 43]
        m = mean_acc(diag, shared)
        vm = mean_acc(van_pw, shared)  # vanilla on THIS method's shared weeks (fair Δ)
        delta = float(m - vm) if (m is not None and vm is not None) else None
        lines.append(f"| {lbl} | {fmt(m)} | {fmt(delta, plus=True)} | "
                     f"{fmt(mean_f1(diag, shared))} | {fmt(mean_acc(diag, far))} |")
    lines.append("")
    lines.append("_Even with a dedicated per-target-week aligned model, the forward-transfer "
                 "gain over the frozen source is negligible — global domain alignment "
                 "(adversarial DANN or covariance-matching CORAL) does not recover the "
                 "per-class discrete drift._")
    return "\n".join(lines)


def main():
    print("# UDA/TTA Benchmark — regenerated from results/inference_auditfix/")
    print()
    print(f"_Source dir: {INFER_DIR}_")
    print(
        "_Forward-only evaluation (source week and later). Mean acc / Macro-F1 over "
        "evaluated weeks; In-dist = source-week test accuracy; Far ≥43 = mean acc weeks "
        "43–52; Δ = mean-acc minus source-only (vanilla) on shared weeks._"
    )
    print()
    if not os.path.isdir(INFER_DIR):
        print(f"ERROR: inference dir not found: {INFER_DIR}", file=sys.stderr)
        sys.exit(1)
    for block in BLOCKS:
        print(render_block(block))
        print()
    uda = render_uda_diagonal()
    if uda:
        print(uda)
        print()


if __name__ == "__main__":
    main()
