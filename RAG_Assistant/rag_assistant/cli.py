"""Command-line interface: ``python -m rag_assistant <command>``."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from .config import BEST_CONFIG, CORPUS_DIR, GATES, GOLDEN_SET, REPORTS_DIR, load_env_file
from .ingest import load_corpus
from .llm import LLMError, OpenAICompatClient, make_llm
from .pipeline import PipelineConfig, RAGPipeline, load_best_config


def _build_pipeline(args) -> RAGPipeline:
    cfg = load_best_config() if getattr(args, "config", "best") == "best" else PipelineConfig()
    if getattr(args, "retriever", None):
        cfg = cfg.with_(retriever=args.retriever)
    if getattr(args, "top_k", None):
        cfg = cfg.with_(top_k=args.top_k)
    llm = None
    if getattr(args, "generator", "extractive") == "llm":
        llm = make_llm(getattr(args, "provider", None), getattr(args, "model", None))
        if llm is None:
            sys.exit("No LLM provider configured. Set MODELSCOPE_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY (see README).")
        cfg = cfg.with_(generator="llm")
    return RAGPipeline.from_corpus(args.corpus, cfg, llm)


def _print_answer(ans, show_context: bool) -> None:
    print(("REFUSED: " if ans.abstained else "") + ans.text)
    if ans.meta.get("error"):
        print(f"  (LLM error, failed closed: {ans.meta['error']})")
    if ans.citations:
        print("  sources: " + ", ".join(ans.citations))
    print(f"  [{ans.generator} | confidence {ans.confidence:.2f} | {ans.latency_ms:.0f} ms]")
    if show_context:
        for h in ans.contexts:
            print(f"    #{h.rank} ({h.score:.3f}) {h.chunk.doc_title} > {h.chunk.heading}: {h.chunk.text[:110]}...")


def cmd_ask(args) -> int:
    _print_answer(_build_pipeline(args).ask(args.question), args.show_context)
    return 0


def cmd_chat(args) -> int:
    pipe = _build_pipeline(args)
    print(f"Halcyon Ridge handbook assistant ({pipe.generator.name}). Empty line to quit.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not q:
            return 0
        _print_answer(pipe.ask(q), args.show_context)


def cmd_eval(args) -> int:
    from .evaluation import evaluate, load_golden
    from .evaluation.gate import check_gates, load_gates
    from .evaluation.judge import LLMJudge
    from .evaluation.report import run_summary_markdown, write_rows

    pipe = _build_pipeline(args)
    judge = None
    if args.judge:
        client = make_llm(args.provider, args.judge_model or args.model)
        if client is None:
            sys.exit("--judge needs an LLM provider (see README).")
        judge = LLMJudge(client)
    items = load_golden(args.golden, split=args.split)
    run = evaluate(pipe, items, judge=judge, label=f"{pipe.generator.name} ({args.split})")
    summary = run.summary()

    out = Path(args.out or REPORTS_DIR / f"eval_{args.generator}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(run_summary_markdown(run, f"Evaluation: {pipe.generator.name}"), encoding="utf-8")
    write_rows(run, out / "per_question.csv")
    (out / "metrics.json").write_text(json.dumps({k: v for k, v in summary.items()}, indent=2, default=str), encoding="utf-8")

    print(f"{run.generator} on {len(items)} questions (split={args.split}); config: {pipe.config.label()}")
    for key in ("hit@5", "mrr", "answer_accuracy", "false_refusal_rate", "abstention_accuracy",
                "unfaithful_rate", "hallucination_rate", "balanced_score", "judge_correct", "judge_faithful"):
        if key in summary:
            m = summary[key]
            print(f"  {key:22s} {m['mean']:.3f}  [{m['lo']:.3f}, {m['hi']:.3f}]  n={m['n']}")
    print(f"  wrote {out}/summary.md")
    if args.split in ("all", "dev"):
        print("  note: the selected config was tuned on these questions - use `study` for an unbiased estimate.")
    if args.gate:
        try:
            gates = load_gates(args.gate)
        except (OSError, ValueError) as exc:
            print(f"error: cannot read the quality-gate file {args.gate} ({exc}). Run `python -m rag_assistant study` to create it.",
                  file=sys.stderr)
            return 2
        failures = check_gates(summary, gates)
        if failures:
            print("QUALITY GATE FAILED:\n  - " + "\n  - ".join(failures))
            return 1
        print("quality gate passed")
    return 0


def cmd_study(args) -> int:
    from .evaluation import load_golden
    from .evaluation.ablation import run_study
    from .evaluation.gate import make_gates
    from .evaluation.report import build_report

    sections = load_corpus(args.corpus)
    items = load_golden(args.golden, split="all")
    study = run_study(sections, items, progress=lambda m: print(m, flush=True))
    out = Path(args.out)
    llm_summary = None
    llm_md = REPORTS_DIR / "eval_llm" / "summary.md"
    if llm_md.exists():
        llm_summary = "See [`../eval_llm/summary.md`](../eval_llm/summary.md) for the LLM run on the same questions."
    report = build_report(study, sections, items, out, llm_summary)
    (out / "study.pkl").write_bytes(pickle.dumps(study))  # dev cache for `report`; git-ignored
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_CONFIG.write_text(json.dumps(study.best.to_dict(), indent=2), encoding="utf-8")
    GATES.write_text(json.dumps(make_gates(study.in_sample.summary(n_boot=0)), indent=2), encoding="utf-8")
    print(f"wrote {report}\nwrote {BEST_CONFIG}\nwrote {GATES}")
    return 0


def cmd_report(args) -> int:
    """Re-render the report and figures from the last `study` run without recomputing it."""
    from .evaluation import load_golden
    from .evaluation.report import build_report

    out = Path(args.out)
    cache = out / "study.pkl"
    if not cache.exists():
        print(f"no cached study at {cache}; run `python -m rag_assistant study` first", file=sys.stderr)
        return 2
    study = pickle.loads(cache.read_bytes())
    print(f"wrote {build_report(study, load_corpus(args.corpus), load_golden(args.golden, split='all'), out)}")
    return 0


def cmd_check_data(args) -> int:
    from .evaluation import check_integrity, load_golden

    problems = check_integrity(load_golden(args.golden, split="all"), load_corpus(args.corpus))
    if problems:
        print("\n".join(f"- {p}" for p in problems))
        return 1
    print("golden set is consistent with the corpus")
    return 0


def cmd_calibrate(args) -> int:
    from .evaluation.calibration import calibrate

    c = calibrate(threshold=args.threshold)
    print(f"faithfulness scorer vs hand labels (n={c.n}, threshold={c.threshold}): accuracy {c.accuracy:.2f}, "
          f"precision {c.unfaithful_precision:.2f}, recall {c.unfaithful_recall:.2f}")
    print("false alarms:", ", ".join(c.false_alarms) or "none")
    print("misses:", ", ".join(c.misses) or "none")
    return 1 if c.unexpected else 0


def cmd_llm_models(args) -> int:
    client = make_llm(args.provider, None)
    if not isinstance(client, OpenAICompatClient):
        sys.exit("`llm-models` lists models for OpenAI-compatible providers (modelscope, openai).")
    try:
        for name in client.list_models():
            print(name)
    except LLMError as exc:
        sys.exit(str(exc))
    return 0


def cmd_fetch_embeddings(args) -> int:
    from .embeddings import ensure_glove

    print(ensure_glove(download=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(prog="rag_assistant", description=__doc__)
    parser.add_argument("--corpus", default=str(CORPUS_DIR), help="folder of .md/.txt/.pdf documents")
    parser.add_argument("--golden", default=str(GOLDEN_SET), help="golden question set (YAML)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_gen(p):
        p.add_argument("--generator", choices=["extractive", "llm"], default="extractive")
        p.add_argument("--provider", choices=["anthropic", "modelscope", "openai"], help="default: auto-detect from API keys")
        p.add_argument("--model", help="LLM model id (default per provider; env RAG_LLM_MODEL)")
        p.add_argument("--retriever", help="override, e.g. bm25, hybrid:bm25+glove")
        p.add_argument("--top-k", type=int, dest="top_k")
        p.add_argument("--config", choices=["best", "default"], default="best")
        p.add_argument("--show-context", action="store_true")

    p = sub.add_parser("ask", help="ask one question"); p.add_argument("question"); add_gen(p); p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("chat", help="interactive Q&A"); add_gen(p); p.set_defaults(fn=cmd_chat)
    p = sub.add_parser("eval", help="evaluate a pipeline on the golden set"); add_gen(p)
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--judge", action="store_true", help="also grade with an LLM judge")
    p.add_argument("--judge-model")
    p.add_argument("--out"); p.add_argument("--gate", nargs="?", const=str(GATES), help="fail if metrics regress (default gates file)")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("study", help="nested cross-validation, ablations and the full report")
    p.add_argument("--out", default=str(REPORTS_DIR / "latest")); p.set_defaults(fn=cmd_study)
    p = sub.add_parser("report", help="re-render reports/latest from the cached study (no recomputation)")
    p.add_argument("--out", default=str(REPORTS_DIR / "latest")); p.set_defaults(fn=cmd_report)
    sub.add_parser("check-data", help="verify the golden set against the corpus").set_defaults(fn=cmd_check_data)
    p = sub.add_parser("calibrate", help="measure the faithfulness scorer against hand labels")
    p.add_argument("--threshold", type=float, default=0.75); p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("llm-models", help="list models your OpenAI-compatible provider offers")
    p.add_argument("--provider", choices=["modelscope", "openai"]); p.set_defaults(fn=cmd_llm_models)
    sub.add_parser("fetch-embeddings", help="download GloVe vectors for the semantic retriever").set_defaults(fn=cmd_fetch_embeddings)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
