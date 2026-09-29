"""Ablations, tuning and nested cross-validation.

Protocol: with only ~90 labelled questions a fixed dev/test split is too noisy (thresholds tuned on 13
unanswerable dev questions looked great on dev and poor on test). So the headline numbers come from
NESTED 5-fold cross-validation: inside each fold, every selection (chunking, retriever, top-k and the
abstention thresholds) is made on the training folds only, and the held-out fold is scored. Pooling the
held-out predictions scores every question with a configuration that never saw it.

Ablation tables that involve no tuning (retriever/chunking comparisons) are descriptive and use all questions.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..embeddings import glove_available
from ..ingest import ChunkConfig, Section
from ..pipeline import PipelineConfig, RAGPipeline
from .calibration import CalibrationResult, SensitivityResult, calibrate, sensitivity
from .dataset import EvalItem
from .harness import EvalRun, evaluate

CHUNK_GRID = {
    "fixed-120w, no heading": ChunkConfig("fixed", 120, 20, add_heading=False),
    "fixed-120w + heading": ChunkConfig("fixed", 120, 20, add_heading=True),
    "structure-60w": ChunkConfig("structure", 60, 1),
    "structure-90w": ChunkConfig("structure", 90, 1),
    "structure-140w": ChunkConfig("structure", 140, 1),
    "structure-90w, no heading": ChunkConfig("structure", 90, 1, add_heading=False),
}
DEFAULT_CHUNK = "structure-90w"

TOP_KS = (3, 5, 8)
COVERAGES = tuple(round(x * 0.1, 1) for x in range(0, 11))  # 0.0 .. 1.0
PRIMARIES = (0.0, 0.3, 0.5)
N_FOLDS = 5

ROW_KEYS = ("hit@1", "hit@3", "hit@5", "hit@8", "recall@5", "mrr", "ndcg@5", "answer_accuracy",
            "false_refusal_rate", "abstention_accuracy", "unfaithful_rate", "hallucination_rate",
            "balanced_score", "token_f1", "citation_precision")


def retriever_grid() -> tuple[str, ...]:
    grid = ["random", "bm25", "tfidf", "lsa", "hybrid:bm25+tfidf", "hybrid:bm25+lsa"]
    if glove_available():  # dense-ish retrievers only when the GloVe vectors are present
        grid += ["glove", "hybrid:bm25+glove", "hybrid:bm25+lsa+glove"]
    return tuple(grid)


def _row(run: EvalRun, **extra) -> dict:
    s = run.summary(n_boot=0)
    return {**extra, **{k: s[k]["mean"] for k in ROW_KEYS if k in s}}


def paired_delta(a: EvalRun, b: EvalRun, metric: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean per-question difference (b - a) with a paired-bootstrap 95% CI, over questions where both have the metric."""
    da = {r.item.id: r.metrics[metric] for r in a.results if metric in r.metrics}
    diffs = np.array([r.metrics[metric] - da[r.item.id] for r in b.results if metric in r.metrics and r.item.id in da])
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(diffs), (n_boot, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def retrieval_ablation(sections: list[Section], items: list[EvalItem], top_k: int = 5,
                       retrievers: tuple[str, ...] | None = None,
                       chunks: dict[str, ChunkConfig] = CHUNK_GRID) -> list[dict]:
    retrievers = retrievers or retriever_grid()
    rows = []
    for cname, ccfg in chunks.items():
        for ret in retrievers:
            if ret == "random" and cname != DEFAULT_CHUNK:
                continue  # one chance-level reference row is enough
            cfg = PipelineConfig(chunk=ccfg, retriever=ret, top_k=top_k)
            rows.append(_row(evaluate(RAGPipeline(sections, cfg), items), retriever=ret, chunking=cname))
    return rows


def select_retrieval(rows_dev: list[dict]) -> tuple[str, str]:
    """Best (retriever, chunking) on dev by MRR, then recall@5. MRR is continuous, so it breaks the
    many recall@5 ties that a corpus this small produces. The random baseline is excluded."""
    best = max((r for r in rows_dev if r["retriever"] != "random"), key=lambda r: (round(r["mrr"], 6), r["recall@5"]))
    return best["retriever"], best["chunking"]


def tune_generator(sections: list[Section], items: list[EvalItem], retriever: str, chunk: ChunkConfig,
                   top_ks=TOP_KS, coverages=COVERAGES, primaries=PRIMARIES) -> tuple[PipelineConfig, list[dict]]:
    """Grid-search top-k and the two abstention thresholds on dev; objective = balanced score.

    Ties are broken towards lower hallucination, then towards the higher (safer) threshold.
    """
    rows = []
    for k, cov, prim in itertools.product(top_ks, coverages, primaries):
        cfg = PipelineConfig(chunk=chunk, retriever=retriever, top_k=k, min_coverage=cov, min_primary=prim)
        rows.append(_row(evaluate(RAGPipeline(sections, cfg), items), top_k=k, min_coverage=cov, min_primary=prim))
    best = max(rows, key=lambda r: (round(r["balanced_score"], 6), -r["hallucination_rate"], r["min_coverage"], r["min_primary"]))
    cfg = PipelineConfig(chunk=chunk, retriever=retriever, top_k=best["top_k"],
                         min_coverage=best["min_coverage"], min_primary=best["min_primary"])
    return cfg, rows


def threshold_sweep(sections: list[Section], items: list[EvalItem], cfg: PipelineConfig,
                    coverages=COVERAGES) -> list[dict]:
    """Vary only min_coverage (other settings fixed) to expose the answer-vs-abstain trade-off."""
    return [_row(evaluate(RAGPipeline(sections, cfg.with_(min_coverage=c)), items), min_coverage=c) for c in coverages]


LADDER_LABELS = (
    "1. Naive RAG: fixed 120-word chunks, BM25, top-3, always answers",
    "2. + chunking chosen by the procedure",
    "3. + retriever and top-k chosen by the procedure",
    "4. + abstention thresholds tuned",
    "5. + answer-type gate = final system",
)


def component_ladder(sections: list[Section], items: list[EvalItem], best: PipelineConfig) -> dict[str, EvalRun]:
    """What does each design decision buy? Cumulative steps from a naive RAG to the tuned system."""
    naive = PipelineConfig(chunk=CHUNK_GRID["fixed-120w, no heading"], retriever="bm25", top_k=3,
                           min_coverage=0.0, min_primary=0.0, heading_weight=1.0, type_gate=False)
    configs = [
        naive,
        naive.with_(chunk=best.chunk),
        naive.with_(chunk=best.chunk, retriever=best.retriever, top_k=best.top_k),
        best.with_(type_gate=False),
        best,
    ]
    return {label: evaluate(RAGPipeline(sections, cfg), items, label=label) for label, cfg in zip(LADDER_LABELS, configs)}


def select_config(sections: list[Section], train: list[EvalItem]) -> tuple[PipelineConfig, list[dict], list[dict]]:
    """The full selection procedure that cross-validation evaluates. Returns (config, retrieval rows, tuning rows)."""
    retrieval = retrieval_ablation(sections, train)
    retriever, chunk_name = select_retrieval(retrieval)
    cfg, tuning = tune_generator(sections, train, retriever, CHUNK_GRID[chunk_name])
    return cfg, retrieval, tuning


def assign_folds(items: list[EvalItem], k: int = N_FOLDS) -> dict[str, int]:
    """Deterministic folds, stratified by question type (the i-th item of each type goes to fold i % k)."""
    counter: dict[str, int] = {}
    folds = {}
    for it in items:
        i = counter.get(it.type, 0)
        counter[it.type] = i + 1
        folds[it.id] = i % k
    return folds


@dataclass
class Study:
    best: PipelineConfig  # selected on ALL questions; what the app ships with
    fold_configs: list[PipelineConfig]  # what each CV fold selected (stability check)
    oof: EvalRun  # pooled out-of-fold results: the honest headline
    in_sample: EvalRun  # `best` scored on the questions it was tuned on (optimistic; for regression gates)
    ladder_oof: dict[str, EvalRun]
    floor: EvalRun  # random retrieval, same generator
    retrieval_table: list[dict]  # all questions, descriptive
    retrieval_by_type: list[dict]
    tuning_table: list[dict]
    sweep: list[dict]
    calibration: CalibrationResult | None = None
    calibration_curve: list[CalibrationResult] = field(default_factory=list)
    sensitivity: SensitivityResult | None = None
    n_questions: int = 0


def merge_runs(runs: list[EvalRun], label: str) -> EvalRun:
    out = EvalRun(label, runs[0].generator, runs[0].ks)
    for r in runs:
        out.results.extend(r.results)
    return out


def retrieval_by_type(sections: list[Section], items: list[EvalItem], chunk: ChunkConfig,
                      retrievers: tuple[str, ...]) -> list[dict]:
    rows = []
    for ret in retrievers:
        run = evaluate(RAGPipeline(sections, PipelineConfig(chunk=chunk, retriever=ret)), [i for i in items if i.answerable])
        for qtype, sub in run.by_type().items():
            s = sub.summary(n_boot=0)
            rows.append({"retriever": ret, "type": qtype, "n": len(sub.results),
                         "recall@5": s["recall@5"]["mean"], "mrr": s["mrr"]["mean"], "hit@1": s["hit@1"]["mean"]})
    return rows


def run_study(sections: list[Section], items: list[EvalItem], k: int = N_FOLDS, progress=None) -> Study:
    say = progress or (lambda msg: None)
    folds = assign_folds(items, k)
    oof_runs, ladder_runs, fold_cfgs = [], [[] for _ in LADDER_LABELS], []
    for f in range(k):
        train = [i for i in items if folds[i.id] != f]
        held = [i for i in items if folds[i.id] == f]
        cfg, _, _ = select_config(sections, train)
        fold_cfgs.append(cfg)
        say(f"fold {f + 1}/{k}: selected {cfg.label()} cov={cfg.min_coverage} primary={cfg.min_primary}")
        oof_runs.append(evaluate(RAGPipeline(sections, cfg), held))
        for j, run in enumerate(component_ladder(sections, held, cfg).values()):
            ladder_runs[j].append(run)

    say("selecting final configuration on all questions")
    best, retrieval_rows, tuning_rows = select_config(sections, items)
    retrievers = retriever_grid()
    return Study(
        best=best,
        fold_configs=fold_cfgs,
        oof=merge_runs(oof_runs, "cross-validated (out-of-fold)"),
        in_sample=evaluate(RAGPipeline(sections, best), items, label="in-sample"),
        ladder_oof={lab: merge_runs(runs, lab) for lab, runs in zip(LADDER_LABELS, ladder_runs)},
        floor=evaluate(RAGPipeline(sections, best.with_(retriever="random")), items, label="random-retrieval floor"),
        retrieval_table=retrieval_rows,
        retrieval_by_type=retrieval_by_type(sections, items, CHUNK_GRID[DEFAULT_CHUNK], tuple(r for r in retrievers if r != "random")),
        tuning_table=tuning_rows,
        sweep=threshold_sweep(sections, items, best, coverages=tuple(round(x * 0.05, 2) for x in range(0, 21))),
        calibration=calibrate(),
        calibration_curve=[calibrate(threshold=t) for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)],
        sensitivity=sensitivity(RAGPipeline(sections, best.with_(min_coverage=0.3, min_primary=0.0)), items, rate=0.5)[0],
        n_questions=len(items),
    )
