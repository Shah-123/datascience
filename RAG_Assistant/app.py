"""Streamlit app: ask the handbook, see the sources, and see how the system itself is evaluated."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_assistant import PipelineConfig, RAGPipeline
from rag_assistant.config import CORPUS_DIR, REPORTS_DIR, load_env_file
from rag_assistant.embeddings import glove_available
from rag_assistant.evaluation.support import score_answer
from rag_assistant.ingest import load_corpus
from rag_assistant.llm import LLMError, detect_provider, make_llm
from rag_assistant.pipeline import load_best_config
from rag_assistant.types import REFUSAL_TEXT

load_env_file()
st.set_page_config(page_title="Handbook Assistant", page_icon="📘", layout="wide")

EXAMPLES = [
    "What is the late payment fee?",
    "How do I get a refund if I drop a class in week 4?",
    "Until which week can a graduate student withdraw with a W?",
    "How often must I change my campus network login secret?",
    "Does the Student Health Center offer dental care?",
    "Who won the 2022 FIFA World Cup?",
]
REPORT_DIR = REPORTS_DIR / "latest"


@st.cache_data
def corpus():
    return load_corpus(CORPUS_DIR)


@st.cache_resource(show_spinner="Building the index...")
def get_pipeline(retriever: str, top_k: int, coverage: float, use_llm: bool) -> RAGPipeline:
    cfg = load_best_config().with_(retriever=retriever, top_k=top_k, min_coverage=coverage,
                                   generator="llm" if use_llm else "extractive")
    return RAGPipeline(corpus(), cfg, make_llm() if use_llm else None)


def retriever_options() -> list[str]:
    opts = ["bm25", "tfidf", "hybrid:bm25+lsa"]
    if glove_available():
        opts += ["glove", "hybrid:bm25+glove"]
    return opts


best = load_best_config()
provider = detect_provider()

with st.sidebar:
    st.header("Settings")
    use_llm = st.radio("Answer generator", ["Offline (extractive)", "LLM"], index=0,
                       help="Extractive quotes the handbook and refuses when unsure. The LLM writes a grounded, cited answer.",
                       ) == "LLM"
    if use_llm and not provider:
        st.error("No LLM provider is configured. Set MODELSCOPE_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY (see README).")
        use_llm = False
    elif use_llm:
        st.caption(f"Provider: {provider}")
    opts = retriever_options()
    retriever = st.selectbox("Retriever", opts, index=opts.index(best.retriever) if best.retriever in opts else 0)
    top_k = st.slider("Passages retrieved (top-k)", 1, 10, best.top_k)
    coverage = st.slider("Abstention threshold", 0.0, 1.0, float(best.min_coverage), 0.05, disabled=use_llm,
                         help="Extractive generator only: the share of the question's key terms that must be found before it answers.")
    st.divider()
    st.caption("Corpus: a **synthetic** university handbook (Halcyon Ridge University). Every fact is invented, so an "
               "answer can only come from retrieval, never from the model's memory.")

st.title("📘 Handbook Assistant")
st.caption("Retrieval-augmented Q&A with citations, refusal when the handbook is silent, and a live faithfulness check.")
tab_ask, tab_eval, tab_corpus = st.tabs(["Ask", "How well does it work?", "Corpus"])

# ----------------------------------------------------------------------------------- ask
with tab_ask:
    st.write("**Try an example**")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state["question"] = ex
    with st.form("ask_form"):
        question = st.text_input("Your question", key="question", placeholder="e.g. How much is the housing deposit?")
        submitted = st.form_submit_button("Ask", type="primary")

    if question.strip():
        try:
            pipe = get_pipeline(retriever, top_k, coverage, use_llm)
            ans = pipe.ask(question.strip())
        except (LLMError, FileNotFoundError, ValueError) as exc:
            st.error(f"Could not answer: {exc}")
            st.stop()

        if ans.abstained:
            st.warning(f"**{REFUSAL_TEXT}**  \nThe assistant refuses rather than guess. This is deliberate: a wrong answer about "
                       "fees or deadlines is worse than no answer.")
            if ans.meta.get("error"):
                st.caption(f"The LLM call failed, so the assistant failed closed: {ans.meta['error']}")
        else:
            with st.container(border=True):
                st.markdown(ans.text)
            faith = score_answer(ans.text, [h.chunk.text for h in ans.contexts])
            if faith.unfaithful:
                st.error("**Faithfulness check: possible unsupported claim.** Verify against the sources below.")
            else:
                st.success("**Faithfulness check passed:** every claim is supported by the retrieved passages.")
            with st.expander("Claim-by-claim check"):
                for c in faith.claims:
                    st.markdown(f"{'✅' if c.supported else '⚠️'} {c.claim}" + (f"  \n<small>{c.reason}</small>" if c.reason else ""),
                                unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Latency", f"{ans.latency_ms:.0f} ms")
        m2.metric("Question-term coverage" if not use_llm else "Generator", f"{ans.confidence:.2f}" if not use_llm else pipe.generator.name.split('[')[-1].rstrip(']'))
        m3.metric("Passages shown to generator", len(ans.contexts))

        if ans.citations:
            st.subheader("Sources cited")
            by_id = {h.chunk.chunk_id: h for h in ans.contexts}
            for cid in ans.citations:
                h = by_id[cid]
                with st.expander(f"{h.chunk.doc_title} > {h.chunk.heading}", expanded=True):
                    st.write(h.chunk.text)
        with st.expander("All retrieved passages"):
            st.dataframe(pd.DataFrame([{
                "rank": h.rank, "score": round(h.score, 4), "document": h.chunk.doc_title, "section": h.chunk.heading,
                "cited": "✓" if h.chunk.chunk_id in ans.citations else "", "text": h.chunk.text} for h in ans.contexts]),
                hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------------- evaluation
with tab_eval:
    metrics_path = REPORT_DIR / "metrics.json"
    if not metrics_path.exists():
        st.info("No evaluation report yet. Run `python -m rag_assistant study` to generate `reports/latest/`.")
    else:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        cv = data["cross_validated"]
        st.write(f"Nested 5-fold cross-validation on **{data['n_questions']} questions** ({data['n_answerable']} answerable). "
                 "Each question is scored by a configuration that never saw it. Intervals are 95% bootstrap CIs.")

        def tile(col, label, key, good_high=True):
            m = cv[key]
            col.metric(label, f"{m['mean']:.0%}", help=f"95% CI {m['lo']:.0%} - {m['hi']:.0%}")

        c = st.columns(5)
        tile(c[0], "Evidence in top 5", "hit@5")
        tile(c[1], "Answer accuracy", "answer_accuracy")
        tile(c[2], "Refuses when it should", "abstention_accuracy")
        tile(c[3], "Hallucination rate", "hallucination_rate")
        tile(c[4], "Balanced score", "balanced_score")
        cal = data["calibration"]
        st.caption(f"Evaluator checks: faithfulness scorer accuracy vs hand labels {cal['accuracy']:.0%}; "
                   f"fault-injection detection {data['fault_injection']['recall']:.0%}.")

        figs = REPORT_DIR / "figures"
        for name, caption in (("outcomes_by_type.png", "Where questions succeed and fail"),
                              ("component_ladder.png", "What each design decision buys"),
                              ("abstention_tradeoff.png", "The answer-versus-refuse trade-off"),
                              ("retrieval_by_type.png", "Retrieval quality by question type")):
            if (figs / name).exists():
                st.image(str(figs / name), caption=caption, use_container_width=True)
        csv = REPORT_DIR / "per_question.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            st.subheader("Every question")
            outcome = st.multiselect("Filter by outcome", sorted(df["outcome"].unique()))
            st.dataframe(df[df["outcome"].isin(outcome)] if outcome else df, hide_index=True, use_container_width=True)
        report = REPORT_DIR / "report.md"
        if report.exists():
            with st.expander("Full written report"):
                st.markdown(report.read_text(encoding="utf-8").replace("](figures/", "](#)("))

# ----------------------------------------------------------------------------------- corpus
with tab_corpus:
    secs = corpus()
    docs = sorted({s.doc_title for s in secs})
    st.write(f"{len(secs)} sections across {len(docs)} synthetic documents.")
    choice = st.selectbox("Document", docs)
    for s in (s for s in secs if s.doc_title == choice):
        with st.expander(s.heading):
            st.write(s.text)
