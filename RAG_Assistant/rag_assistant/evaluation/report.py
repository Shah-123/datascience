"""Turn a Study (or a single EvalRun) into figures, JSON, CSV and a readable markdown report."""
from __future__ import annotations

import csv
import json
import math
import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..ingest import Section, chunk_sections  # noqa: E402
from .ablation import DEFAULT_CHUNK, LADDER_LABELS, Study, paired_delta  # noqa: E402
from .dataset import TYPES, EvalItem  # noqa: E402
from .harness import EvalRun  # noqa: E402

# Palette: the validated default from the dataviz method (light surface). Slots are assigned in fixed order.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"

TYPE_LABELS = {
    "factual": "Factual", "paraphrase": "Paraphrase", "conditional": "Conditional", "multi_hop": "Multi-hop",
    "unanswerable_near": "Near-miss (unanswerable)", "unanswerable_off": "Off-topic (unanswerable)",
}
# Stack order = palette slot order (fixed). Green is kept for the benign outcome, never for a failure.
OUTCOME_GROUPS = [
    ("Correct", BLUE, ("correct", "correct_abstention")),
    ("Wrong answer: evidence was retrieved", ORANGE, ("wrong_answer_generation",)),
    ("Wrong answer: retrieval missed the evidence", AQUA, ("wrong_answer_retrieval",)),
    ("Answered an unanswerable question", YELLOW, ("over_answer",)),
    ("Refused: evidence was retrieved", MAGENTA, ("false_refusal_evidence_present",)),
    ("Refused: retrieval missed the evidence", GREEN, ("false_refusal_retrieval",)),
]


# ------------------------------------------------------------------------------- formatting


def _f(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def ci(m: dict | None, nd: int = 2) -> str:
    if not m:
        return "n/a"
    if math.isnan(m.get("lo", float("nan"))):
        return _f(m["mean"], nd)
    return f"{m['mean']:.{nd}f} ({m['lo']:.{nd}f}-{m['hi']:.{nd}f})"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _clean(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else round(obj, 5)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


# ------------------------------------------------------------------------------- figures


def _base(figsize=(8.2, 4.4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1, color=AXIS)
    ax.grid(color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    return fig, ax


def _titles(fig, title: str, subtitle: str, top: float = 0.965):
    fig.text(0.012, top, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, top - 0.062, subtitle, fontsize=9.5, color=INK2, va="top")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_retrieval_by_type(study: Study, path: Path) -> None:
    rows = study.retrieval_by_type
    rets = sorted({r["retriever"] for r in rows})
    para = {r["retriever"]: r["mrr"] for r in rows if r["type"] == "paraphrase"}
    other = {}
    for ret in rets:
        sub = [r for r in rows if r["retriever"] == ret and r["type"] != "paraphrase"]
        other[ret] = sum(r["mrr"] * r["n"] for r in sub) / sum(r["n"] for r in sub)
    order = sorted(rets, key=lambda r: para[r])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), facecolor=SURFACE, sharey=True)
    for ax, data, name in ((axes[0], other, "Factual, conditional and multi-hop questions"), (axes[1], para, "Paraphrased questions")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        vals = [data[r] for r in order]
        ax.barh(range(len(order)), vals, height=0.42, color=BLUE)
        for i, v in enumerate(vals):
            ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=9, color=INK2)
        ax.set_xlim(0, 1.12)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([r.replace("hybrid:", "hybrid ") for r in order], color=INK2, fontsize=9)
        ax.set_title(name, fontsize=10, color=INK2, loc="left", pad=8)
        ax.set_xlabel("Mean reciprocal rank (higher is better)", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.74, bottom=0.16, wspace=0.08)
    title, sub = retrieval_story(para, other)
    _titles(fig, title, sub)
    _save(fig, path)


def retrieval_story(para: dict, other: dict) -> tuple[str, str]:
    """A title that states what the data actually shows (computed, so it cannot go stale)."""
    sub = "Mean reciprocal rank of the first chunk containing the gold evidence (answerable questions, default chunking)."
    dense = [r for r in para if "glove" in r and r != "glove"]
    if "bm25" in para and dense:
        best = max(dense, key=lambda r: para[r])
        gain, loss = para[best] - para["bm25"], other["bm25"] - other[best]
        if gain > 0.05 and loss > 0.02:
            return "Embeddings help paraphrased questions and slightly hurt the rest", sub
        if gain > 0.05:
            return "Embeddings help paraphrased questions without hurting the rest", sub
    spread = max(para.values()) - min(para.values())
    return ("Retrievers differ mainly on paraphrased questions" if spread > 0.1 else "Retrievers perform alike on this corpus"), sub


def fig_tradeoff(study: Study, path: Path) -> None:
    sw = study.sweep
    xs = [r["min_coverage"] for r in sw]
    fig, ax = _base((8.2, 4.3))
    fig.subplots_adjust(left=0.09, right=0.79, top=0.77, bottom=0.25)
    series = [("answer_accuracy", "Answers correctly\n(answerable)", BLUE), ("abstention_accuracy", "Refuses correctly\n(unanswerable)", ORANGE)]
    for key, label, colour in series:
        ys = [r[key] for r in sw]
        ax.plot(xs, ys, color=colour, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label.replace("\n", " "))
        ax.text(xs[-1] + 0.015, ys[-1], label, color=INK2, fontsize=9, va="center")
    tau = study.best.min_coverage
    ax.axvline(tau, color=AXIS, linewidth=1)
    ax.text(tau + 0.01, 1.035, f"selected: {tau:g}", color=INK2, fontsize=9, va="bottom")
    for key, _, colour in series:
        y = next(r[key] for r in sw if abs(r["min_coverage"] - tau) < 1e-9) if any(abs(r["min_coverage"] - tau) < 1e-9 for r in sw) else None
        if y is not None:
            ax.scatter([tau], [y], s=64, color=colour, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Abstention threshold: minimum question-term coverage required to answer", fontsize=9, color=MUTED)
    ax.set_ylabel("Share of questions", fontsize=9, color=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.44, 0.0))
    _titles(fig, "Caution is paid for in correct answers",
            "Offline extractive generator, all questions, other settings fixed. There is no free lunch on this curve.")
    _save(fig, path)


def fig_outcomes(run: EvalRun, path: Path) -> None:
    by_type = run.by_type()
    types = [t for t in TYPES if t in by_type]
    fig, ax = _base((9.0, 4.6))
    fig.subplots_adjust(left=0.29, right=0.985, top=0.76, bottom=0.27)
    ax.grid(axis="y", visible=False)
    for i, t in enumerate(types):
        counts = by_type[t].outcomes()
        total = sum(counts.values())
        left = 0.0
        for name, colour, keys in OUTCOME_GROUPS:
            n = sum(counts.get(k, 0) for k in keys)
            if not n:
                continue
            w = n / total
            ax.barh(i, w, left=left, height=0.5, color=colour, edgecolor=SURFACE, linewidth=2)
            if w >= 0.09:  # only label a segment when the text clearly fits
                ax.text(left + w / 2, i, str(n), ha="center", va="center", fontsize=9,
                        color="white" if colour in (BLUE, GREEN, ORANGE) else INK)
            left += w
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{TYPE_LABELS[t]} (n={len(by_type[t].results)})" for t in types], color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in OUTCOME_GROUPS]
    ax.legend(handles, [n for n, _, _ in OUTCOME_GROUPS], loc="upper center", bbox_to_anchor=(0.36, -0.11),
              ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.1, columnspacing=1.6)
    _titles(fig, "Most failures are refusals of answerable questions, not made-up answers",
            "Outcome of every question in the cross-validated run, by question type (counts inside segments).")
    _save(fig, path)


def fig_ladder(study: Study, path: Path) -> None:
    runs = list(study.ladder_oof.items())
    fig, ax = _base((9.0, 4.3))
    fig.subplots_adjust(left=0.40, right=0.97, top=0.79, bottom=0.14)
    ax.grid(axis="y", visible=False)
    for i, (label, run) in enumerate(runs):
        m = run.summary(n_boot=1000)["balanced_score"]
        colour = BLUE if i == len(runs) - 1 else MUTED
        ax.plot([m["lo"], m["hi"]], [i, i], color=colour, linewidth=2, solid_capstyle="round")
        ax.scatter([m["mean"]], [i], s=70, color=colour, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.text(min(m["hi"] + 0.015, 0.95), i - 0.3, f"{m['mean']:.2f}", fontsize=9, color=INK2, va="center")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([textwrap.fill(lab, 46) for lab, _ in runs], color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced score (dot = mean, line = 95% CI)", fontsize=9, color=MUTED)
    _titles(fig, "What each design decision buys (cross-validated)",
            "Cumulative steps from a naive RAG to the final system; the last step is highlighted.")
    _save(fig, path)


# ------------------------------------------------------------------------------- report body


def _headline_rows(summary: dict) -> list[list[str]]:
    spec = [
        ("Retrieval: Hit@1", "hit@1", "Correct evidence is the top-ranked chunk"),
        ("Retrieval: Hit@5", "hit@5", "Evidence appears somewhere in the top 5"),
        ("Retrieval: Recall@5", "recall@5", "Share of required evidence phrases found in the top 5 (multi-hop needs several)"),
        ("Retrieval: MRR", "mrr", "1 / rank of the first relevant chunk"),
        ("Retrieval: nDCG@5", "ndcg@5", "Ranking quality of the top 5"),
        ("Answer accuracy", "answer_accuracy", "Answerable questions answered with every required fact"),
        ("False-refusal rate", "false_refusal_rate", "Answerable questions the system refused (lower is better)"),
        ("Abstention accuracy", "abstention_accuracy", "Unanswerable questions correctly refused"),
        ("Unfaithful-answer rate", "unfaithful_rate", "Answers with a claim unsupported by the retrieved text (lower is better)"),
        ("Hallucination rate", "hallucination_rate", "All questions: answered AND (unanswerable OR unfaithful) (lower is better)"),
        ("Balanced score", "balanced_score", "Mean of answer accuracy and abstention accuracy"),
        ("Citation precision", "citation_precision", "Cited chunks that really contain the evidence"),
        ("Token F1 vs reference", "token_f1", "Word overlap with the reference answer (answered questions only)"),
    ]
    return [[name, ci(summary.get(key)), desc] for name, key, desc in spec if key in summary]


def _outcome_table(run: EvalRun) -> str:
    by_type = run.by_type()
    headers = ["Question type", "n"] + [n for n, _, _ in OUTCOME_GROUPS]
    rows = []
    for t in TYPES:
        if t not in by_type:
            continue
        c = by_type[t].outcomes()
        rows.append([TYPE_LABELS[t], str(len(by_type[t].results))] + [str(sum(c.get(k, 0) for k in keys)) for _, _, keys in OUTCOME_GROUPS])
    return table(headers, rows)


def _failure_examples(run: EvalRun, limit: int = 10) -> str:
    bad = [r for r in run.results if r.outcome not in ("correct", "correct_abstention")]
    priority = {"over_answer": 0, "wrong_answer_generation": 1, "wrong_answer_retrieval": 2,
                "false_refusal_evidence_present": 3, "false_refusal_retrieval": 4}
    bad.sort(key=lambda r: (priority.get(r.outcome, 9), r.item.id))
    rows = []
    for r in bad[:limit]:
        ans = r.answer.text.replace("|", "/").replace("\n", " ")
        rows.append([r.item.id, r.outcome.replace("_", " "), r.item.question.replace("|", "/"), ans[:110] + ("..." if len(ans) > 110 else "")])
    return table(["id", "outcome", "question", "system answer"], rows)


def ladder_findings(study: Study) -> str:
    """Which steps of the ladder changed anything beyond noise? Computed from paired bootstrap CIs."""
    runs = list(study.ladder_oof.items())
    significant, flat = [], []
    for i in range(1, len(runs)):
        (_, prev), (label, cur) = runs[i - 1], runs[i]
        parts = []
        for metric, name in (("hallucination_rate", "hallucination rate"), ("answer_accuracy", "answer accuracy")):
            d, lo, hi = paired_delta(prev, cur, metric)
            if lo > 0 or hi < 0:
                parts.append(f"{name} {d:+.2f} (paired 95% CI {lo:+.2f} to {hi:+.2f})")
        (significant if parts else flat).append((i + 1, parts))
    lines = [f"- Step {n} changes " + " and ".join(parts) + "." for n, parts in significant]
    if flat:
        lines.append("- Steps " + ", ".join(str(n) for n, _ in flat) + " make no statistically detectable difference to hallucination rate or answer accuracy (paired CIs include 0).")
    return "\n".join(lines)


def retrieval_sentence(study: Study) -> str:
    rows = study.retrieval_by_type
    def stat(ret):
        p = [r for r in rows if r["retriever"] == ret and r["type"] == "paraphrase"]
        o = [r for r in rows if r["retriever"] == ret and r["type"] != "paraphrase"]
        if not p or not o:
            return None
        return p[0]["mrr"], sum(r["mrr"] * r["n"] for r in o) / sum(r["n"] for r in o), p[0]["n"], sum(r["n"] for r in o)
    base = stat("bm25")
    dense = [r for r in {x["retriever"] for x in rows} if "glove" in r and r != "glove"]
    if not base or not dense:
        return ""
    best = max(dense, key=lambda r: stat(r)[0])
    b = stat(best)
    return (f"On the {b[2]} paraphrased questions the best embedding hybrid (`{best}`) reaches MRR {b[0]:.2f} vs {base[0]:.2f} for BM25; "
            f"on the other {b[3]} answerable questions BM25 reaches {base[1]:.2f} vs {b[1]:.2f}. "
            "The gaps are small relative to the sample size, but they are consistent with dense signals helping when the words differ and adding noise when they match.")


def build_report(study: Study, sections: list[Section], items: list[EvalItem], out_dir: Path, llm_summary: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = out_dir / "figures"
    fig_retrieval_by_type(study, figs / "retrieval_by_type.png")
    fig_tradeoff(study, figs / "abstention_tradeoff.png")
    fig_outcomes(study.oof, figs / "outcomes_by_type.png")
    fig_ladder(study, figs / "component_ladder.png")

    oof = study.oof.summary()
    ins = study.in_sample.summary()
    floor = study.floor.summary()
    n_ans = sum(i.answerable for i in items)
    n_chunks = len(chunk_sections(sections, study.best.chunk))
    best = study.best

    # ladder table with paired deltas vs previous step
    ladder_rows, prev = [], None
    for label, run in study.ladder_oof.items():
        s = run.summary(n_boot=1000)
        d_h = d_a = ""
        if prev is not None:
            d, lo, hi = paired_delta(prev, run, "hallucination_rate")
            d_h = f"{d:+.2f} ({lo:+.2f} to {hi:+.2f})"
            d, lo, hi = paired_delta(prev, run, "answer_accuracy")
            d_a = f"{d:+.2f} ({lo:+.2f} to {hi:+.2f})"
        ladder_rows.append([label, ci(s["answer_accuracy"]), ci(s["abstention_accuracy"]), ci(s["hallucination_rate"]), ci(s["balanced_score"]), d_h, d_a])
        prev = run

    # retrieval table (descriptive, all questions): every retriever at the SAME default chunking
    ret_rows = [[r["retriever"], _f(r["hit@1"]), _f(r["hit@5"]), _f(r["recall@5"]), _f(r["mrr"]), _f(r["ndcg@5"])]
                for r in study.retrieval_table if r["chunking"] == DEFAULT_CHUNK]
    ret_rows.sort(key=lambda r: -float(r[4]))

    chunk_rows = []
    for r in study.retrieval_table:
        if r["retriever"] == best.retriever:
            chunk_rows.append([r["chunking"], _f(r["hit@1"]), _f(r["recall@5"]), _f(r["mrr"]), _f(r["ndcg@5"])])

    cal = study.calibration
    curve_rows = [[f"{c.threshold:g}", _f(c.accuracy), _f(c.unfaithful_precision), _f(c.unfaithful_recall)] for c in study.calibration_curve]
    sens = study.sensitivity
    kind_rows = [[k, f"{d}/{n}", _f(d / n)] for k, (d, n) in sorted(sens.by_kind.items())]
    fold_rows = [[str(i + 1), c.retriever, c.chunk.label(), str(c.top_k), f"{c.min_coverage:g}", f"{c.min_primary:g}"] for i, c in enumerate(study.fold_configs)]

    gap = ins["balanced_score"]["mean"] - oof["balanced_score"]["mean"]
    fa = oof["abstention_accuracy"]["mean"]
    llm_block = llm_summary or (
        "**Not run yet.** This report measures the *offline extractive* generator only, because the sandbox that produced it had no "
        "route to an LLM provider. Run `python -m rag_assistant eval --generator llm` with your provider configured "
        "(see the README) and the same harness will report an LLM on the same 88 questions.")

    md = f"""# RAG evaluation report

Halcyon Ridge University handbook (synthetic) - {date.today().isoformat()} - generated by `python -m rag_assistant study`.

**Setup.** {len(sections)} sections in {len({s.doc_id for s in sections})} documents, {n_chunks} chunks with the selected chunking. \
Golden set: **{len(items)} questions** ({n_ans} answerable, {len(items) - n_ans} unanswerable). Generator: offline extractive (no LLM). \
Every number below is either a cross-validated estimate or is labelled otherwise; intervals are 95% bootstrap CIs over questions.

## 1. Headline - cross-validated

Nested 5-fold cross-validation: in each fold the chunking, retriever, top-k and abstention thresholds are chosen on the training folds only, \
and the held-out fold is scored. Each question is answered by a configuration that never saw it.

{table(["Metric", "Value (95% CI)", "Meaning"], _headline_rows(oof))}

Latency (median / p95): {_f(oof['latency_p50_ms']['mean'], 1)} ms / {_f(oof['latency_p95_ms']['mean'], 1)} ms per question, retrieval and generation included.

**How to read this.**
- Retrieval is strong: the gold evidence is in the top 5 for {oof['hit@5']['mean']:.0%} of answerable questions.
- The extractive generator is the bottleneck: it gets {oof['answer_accuracy']['mean']:.0%} of answerable questions right and correctly refuses {fa:.0%} of unanswerable ones.
- Refusals concentrate where the wording differs from the handbook or an answer needs two passages: paraphrase and multi-hop questions are almost all refused (see the table below). \
Retrieval found the evidence for nearly all of them, so this is the gap an LLM generator is meant to close, not a retrieval problem.
- The operating point comes from maximising the balanced score, which weighs refusing correctly as much as answering correctly. If a wrong answer is cheaper than a refusal in your setting, lower the threshold (section 5).
- Scoring the same configuration on the questions it was tuned on gives balanced score {ins['balanced_score']['mean']:.2f} vs the honest {oof['balanced_score']['mean']:.2f}: \
tuning on ~90 questions inflates results by {gap:+.2f}. That is why the headline is cross-validated.
- A retriever that returns random chunks scores balanced {floor['balanced_score']['mean']:.2f}: it "wins" abstention only by refusing almost everything, which is why abstention accuracy is never reported alone.

## 2. Where does it fail?

![Outcomes by question type](figures/outcomes_by_type.png)

{_outcome_table(study.oof)}

Failure taxonomy: a *retrieval miss* means the gold evidence was not in the passages the generator saw; the other failures happened **with the evidence in hand**, \
so they are generator faults. Selected failures (most serious first):

{_failure_examples(study.oof)}

Full per-question results: [`per_question.csv`](per_question.csv).

## 3. What each design decision buys

![Component ladder](figures/component_ladder.png)

{table(["Step", "Answer acc.", "Abstention acc.", "Hallucination rate", "Balanced", "Δ hallucination vs previous step", "Δ answer acc. vs previous step"], ladder_rows)}

Deltas are paired over the same questions with 95% bootstrap CIs. What they show:

{ladder_findings(study)}

## 4. Retrieval ablation (descriptive, all {n_ans} answerable questions)

![Retrieval by question type](figures/retrieval_by_type.png)

All retrievers at the same chunking (`{DEFAULT_CHUNK}`), sorted by MRR:

{table(["Retriever", "Hit@1", "Hit@5", "Recall@5", "MRR", "nDCG@5"], ret_rows)}

{retrieval_sentence(study)}

Chunking comparison for `{best.retriever}`:

{table(["Chunking", "Hit@1", "Recall@5", "MRR", "nDCG@5"], chunk_rows)}

Caveats worth knowing: with only {n_chunks} chunks, top-5 covers about {5 / n_chunks:.0%} of the whole corpus, so Recall@5 saturates; \
differences between retrievers are mostly at rank 1 and on paraphrased questions. `lsa` scores exactly like `tfidf` because \
{n_chunks} chunks is too few for a truncated SVD to be lossy - it adds no semantics here.

## 5. The abstention trade-off

![Abstention trade-off](figures/abstention_tradeoff.png)

Selected configuration (all questions): retriever `{best.retriever}`, chunking `{best.chunk.label()}`, top-{best.top_k}, \
coverage threshold {best.min_coverage:g}, primary-sentence threshold {best.min_primary:g}, answer-type gate {'on' if best.type_gate else 'off'}.

Stability of the selection across the 5 cross-validation folds:

{table(["Fold", "Retriever", "Chunking", "top-k", "Coverage threshold", "Primary threshold"], fold_rows)}

## 6. Can the evaluator be trusted?

**Faithfulness scorer vs hand labels** ({cal.n} labelled answers in `data/eval/support_calibration.yaml`, threshold {cal.threshold:g}): \
accuracy {cal.accuracy:.2f}, precision of the "unfaithful" flag {cal.unfaithful_precision:.2f}, recall {cal.unfaithful_recall:.2f}. \
The default threshold sits on a plateau:

{table(["Threshold", "Accuracy", "Precision (unfaithful)", "Recall (unfaithful)"], curve_rows)}

Known blind spots, kept in the calibration set on purpose - false alarms: {', '.join(cal.false_alarms) or 'none'}; \
misses: {', '.join(cal.misses) or 'none'}. \
A lexical scorer cannot see a synonym paraphrase (false alarm) or a right number attached to the wrong group, a changed unit, \
a flipped quantifier ("at most" vs "at least"), or a short invented clause inside a long sentence (misses). Use the LLM judge (`--judge`) to cover those.

**Fault injection.** {sens.injected} answers were deliberately corrupted (changed number, invented sentence, flipped negation); \
the harness flagged {sens.detected} of them (recall {sens.recall:.2f}) and raised {sens.false_alarms} false alarms on {sens.clean} untouched answers ({sens.false_alarm_rate:.2f}).

{table(["Fault kind", "Detected / injected", "Recall"], kind_rows)}

**Golden-set integrity.** `tests/test_data_integrity.py` verifies that every evidence phrase occurs exactly once in the corpus, \
every required fact is derivable from its evidence, and that the checker itself fails on deliberately broken data.

## 7. LLM generator

{llm_block}

## 8. Limitations

- **Small sample.** {len(items)} questions; intervals are wide and small differences (a few points) are not significant. The paired deltas in section 3 say which changes are.
- **Synthetic, single-author corpus and questions.** Chosen so a model cannot answer from memory; but the wording is mine, so paraphrase difficulty is unrepresentative of real user language. Real deployment needs real questions (log them, label a sample).
- **The generator evaluated here is extractive.** It cannot synthesise or paraphrase, so its high faithfulness is by construction and its accuracy is a floor, not what an LLM would achieve.
- **Lexical faithfulness scoring** has the blind spots listed in section 6.
- **One annotator.** No inter-annotator agreement was measured.

## 9. Reproduce

```bash
pip install -r requirements.txt
python -m rag_assistant fetch-embeddings   # optional: GloVe vectors (134 MB) for the semantic retriever
python -m rag_assistant study              # regenerates this report
```
"""
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    # machine-readable outputs
    payload = {
        "generated": date.today().isoformat(),
        "n_questions": len(items), "n_answerable": n_ans,
        "best_config": {"retriever": best.retriever, "chunk": best.chunk.label(), "top_k": best.top_k,
                        "min_coverage": best.min_coverage, "min_primary": best.min_primary, "type_gate": best.type_gate},
        "cross_validated": oof, "in_sample": ins, "random_retrieval_floor": floor,
        "ladder": {k: v.summary(n_boot=1000) for k, v in study.ladder_oof.items()},
        "calibration": {"accuracy": cal.accuracy, "precision": cal.unfaithful_precision, "recall": cal.unfaithful_recall,
                        "false_alarms": cal.false_alarms, "misses": cal.misses},
        "fault_injection": {"injected": sens.injected, "detected": sens.detected, "recall": sens.recall,
                            "false_alarm_rate": sens.false_alarm_rate},
    }
    (out_dir / "metrics.json").write_text(json.dumps(_clean(payload), indent=2), encoding="utf-8")
    write_rows(study.oof, out_dir / "per_question.csv")
    return out_dir / "report.md"


def write_rows(run: EvalRun, path: Path) -> None:
    rows = run.rows()
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_summary_markdown(run: EvalRun, title: str) -> str:
    s = run.summary()
    out = [f"# {title}", "", f"Generator: `{run.generator}` - {len(run.results)} questions.", "",
           table(["Metric", "Value (95% CI)", "Meaning"], _headline_rows(s)), "", "## Outcomes by question type", "", _outcome_table(run), "",
           "## Selected failures", "", _failure_examples(run)]
    for k, label in (("judge_correct", "LLM-judge correctness"), ("judge_faithful", "LLM-judge faithfulness")):
        if k in s:
            out += ["", f"{label}: {ci(s[k])}"]
    return "\n".join(out) + "\n"
