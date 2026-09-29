import dataclasses

from rag_assistant.evaluation.dataset import TYPES, check_integrity, load_golden


def test_golden_set_is_consistent_with_the_corpus(items, sections):
    assert check_integrity(items, sections) == []


def test_golden_set_shape(items):
    assert len(items) >= 80
    assert {i.type for i in items} == set(TYPES)
    assert sum(i.answerable for i in items) >= 55 and sum(not i.answerable for i in items) >= 20


def test_dev_and_test_splits_are_disjoint_and_cover_every_type():
    dev, test = load_golden(split="dev"), load_golden(split="test")
    assert not {i.id for i in dev} & {i.id for i in test}
    assert {i.type for i in dev} == {i.type for i in test} == set(TYPES)


def test_checker_detects_broken_data(items, sections):
    """A checker that can't fail proves nothing: corrupt the data five different ways."""
    bad = list(items)
    bad[0] = dataclasses.replace(bad[0], evidence=("this phrase is not in the corpus",))
    bad[1] = dataclasses.replace(bad[1], evidence=("cumulative GPA",))  # ambiguous: occurs many times
    bad[2] = dataclasses.replace(bad[2], facts=(("999 weeks",),))  # not derivable from evidence
    bad[3] = dataclasses.replace(bad[3], evidence=())  # answerable without evidence
    bad[-1] = dataclasses.replace(bad[-1], evidence=("Passwords expire every 180 days",))  # unanswerable with evidence
    problems = " ".join(check_integrity(bad, sections))
    for needle in ("not in the corpus", "occurs 8 times", "999 weeks", "need answer, facts and evidence", "must have no answer"):
        assert needle in problems, needle


def test_duplicate_ids_are_reported(items, sections):
    dup = list(items) + [items[0]]
    assert any("duplicate id" in p for p in check_integrity(dup, sections))
