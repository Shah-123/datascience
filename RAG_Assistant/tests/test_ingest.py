import pytest

from rag_assistant.ingest import ChunkConfig, chunk_sections, load_corpus, parse_markdown


def test_corpus_loads_all_documents(sections):
    assert len({s.doc_id for s in sections}) == 10
    assert all(s.text.strip() for s in sections)
    assert len({s.section_id for s in sections}) == len(sections)  # ids are unique


def test_front_matter_is_stripped_and_title_kept(sections):
    assert not any("synthetic: true" in s.text for s in sections)
    assert any(s.doc_title == "Academic Regulations" for s in sections)


def test_markdown_tables_are_linearised(sections):
    grading = next(s for s in sections if s.heading == "Grading Scale")
    assert "Letter grade B-: Grade points 2.7, Percentage 80-82." in grading.text
    assert "|" not in grading.text


def test_structure_chunks_stay_inside_sections_and_respect_size(sections):
    cfg = ChunkConfig("structure", max_words=60, overlap=1)
    for c in chunk_sections(sections, cfg):
        sentences = c.text.split(". ")
        # a chunk may exceed max_words only if it is a single (long) sentence
        assert len(c.text.split()) <= 60 or len(sentences) == 1


def test_structure_chunking_covers_every_sentence(sections):
    from rag_assistant.text import split_sentences
    chunks = chunk_sections(sections, ChunkConfig("structure", 60, 1))
    for sec in sections:
        joined = " ".join(c.text for c in chunks if c.section_id == sec.section_id)
        assert all(s in joined for s in split_sentences(sec.text))


def test_fixed_chunking_overlaps_and_ignores_section_boundaries(sections):
    chunks = chunk_sections(sections, ChunkConfig("fixed", 50, 10, add_heading=False))
    assert all(len(c.text.split()) <= 50 for c in chunks)
    first_doc = [c for c in chunks if c.doc_id == chunks[0].doc_id]
    assert first_doc[0].text.split()[-10:] == first_doc[1].text.split()[:10]


def test_heading_is_prepended_only_when_requested(sections):
    c = chunk_sections(sections, ChunkConfig())[0]
    assert c.heading in c.index_text(True) and c.heading not in c.index_text(False)


def test_unknown_strategy_and_empty_corpus_fail_loudly(tmp_path, sections):
    with pytest.raises(ValueError):
        chunk_sections(sections, ChunkConfig(strategy="nope"))
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path)


def test_plain_text_and_subheadings(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text("# T\n\n## A\nfirst line.\n\n### Sub\n- bullet one.\n- bullet two.\n\n## B\nsecond.\n", encoding="utf-8")
    secs = parse_markdown(md)
    assert [s.heading for s in secs] == ["A", "B"]
    assert "bullet one." in secs[0].text and "- " not in secs[0].text
