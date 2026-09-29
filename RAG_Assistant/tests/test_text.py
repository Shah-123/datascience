from rag_assistant.text import (contains_phrase, content_terms, is_number, normalize_for_match, split_sentences,
                                stem, tokenize)


def test_tokenize_normalises_numbers_and_currency():
    assert tokenize("Tuition is $1,150 per credit, refunded 60%.") == ["tuition", "is", "1150", "per", "credit", "refunded", "60"]
    assert "3.3" in tokenize("GPA of at least 3.3")


def test_tokenize_am_pm_and_number_words():
    assert tokenize("from 6 p.m. to 2 a.m.") == ["from", "6", "pm", "to", "2", "am"]
    assert tokenize("at least three members") == ["at", "least", "3", "members"]


def test_negation_contraction_becomes_not():
    assert "not" in tokenize("Students don't pay")


def test_stemming_conflates_inflections_but_not_numbers():
    assert stem("refunded") == stem("refund") == stem("refunds")
    assert stem("180") == "180" and is_number("3.3") and not is_number("gpa")


def test_content_terms_drop_stopwords_but_can_keep_negations():
    assert "the" not in content_terms("the late fee")
    assert "not" not in content_terms("is not eligible")
    assert "not" in content_terms("is not eligible", keep_negations=True)


def test_sentence_split_respects_decimals_and_abbreviations():
    text = "GPA of 3.3 is required. The library opens at 8 a.m. on weekdays. Fees apply."
    assert split_sentences(text) == ["GPA of 3.3 is required.", "The library opens at 8 a.m. on weekdays.", "Fees apply."]


def test_sentence_split_treats_newlines_as_boundaries():
    assert split_sentences("Row one.\nRow two.") == ["Row one.", "Row two."]


def test_contains_phrase_ignores_case_punctuation_and_formatting():
    assert contains_phrase("A late payment fee of $75 is charged.", "late payment fee of $75")
    assert not contains_phrase("A late payment fee of $75", "fee of $7")
    assert normalize_for_match("Hello,  World!") == " hello world "
