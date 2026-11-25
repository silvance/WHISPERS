import importlib.util

import pytest

from whispers import translation
from whispers.documents import Section
from whispers.translation import detect_language, translate_document


def test_chunk_text_rejoins_and_bounds():
    text = "line one\nline two\n\nparagraph two with more words\n"
    chunks = translation._chunk_text(text, max_chars=10)
    # Concatenation is lossless (structure preserved).
    assert "".join(chunks) == text
    # Every chunk respects the cap.
    assert all(len(c) <= 10 for c in chunks)


def test_chunk_text_hard_splits_long_line():
    text = "x" * 25
    chunks = translation._chunk_text(text, max_chars=10)
    assert "".join(chunks) == text
    assert all(len(c) <= 10 for c in chunks)
    assert len(chunks) == 3


def test_chunk_text_empty():
    assert translation._chunk_text("") == [""]


def test_translate_text_without_argos(monkeypatch):
    # argostranslate is not installed in this environment -> clear RuntimeError.
    with pytest.raises(RuntimeError, match="Argos Translate is not installed"):
        translation.translate_text("hola", from_code="es", to_code="en")


def test_available_source_languages_without_argos():
    with pytest.raises(RuntimeError, match="Argos Translate is not installed"):
        translation.available_source_languages()


def test_translate_files_without_argos(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hola", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Argos Translate is not installed"):
        translation.translate_files([f], from_code="es", to_code="en")


def test_detect_language_empty_is_none():
    assert detect_language("") is None
    assert detect_language("   ") is None


def test_detect_language_without_langdetect():
    if importlib.util.find_spec("langdetect") is not None:
        pytest.skip("langdetect installed; missing-dependency path not exercised")
    assert detect_language("this is plainly english text") is None


def test_detect_language_english():
    pytest.importorskip("langdetect")
    assert detect_language("This is a clearly English sentence about the weather.") == (
        "en"
    )


# -- Documents ---------------------------------------------------------------

NAMES = {"zh": "Chinese", "ar": "Arabic", "ko": "Korean"}


class _Fake:
    """Stands in for an Argos translator and counts what it was given."""

    calls: list = []

    def __init__(self, code):
        self.code = code

    def translate(self, text):
        _Fake.calls.append((self.code, text))
        return f"<{self.code}: {text}>"


def _translate(sections, **kwargs):
    _Fake.calls = []
    kwargs.setdefault("forced", None)
    kwargs.setdefault("installed", {"zh", "ar"})
    return translate_document(
        sections,
        names=NAMES,
        detect=lambda _: [("en", 0.99)],
        translator_for=_Fake,
        **kwargs,
    )


def test_only_foreign_paragraphs_go_to_the_translator():
    section = Section(
        paragraphs=[
            "Meeting notes, prepared by the liaison officer for the section.",
            "会议定于下周三上午九点举行。",
            "Smith, John",
        ]
    )
    document = _translate([section])
    assert _Fake.calls == [("zh", "会议定于下周三上午九点举行。")]
    english, chinese, name = document.sections[0].paragraphs
    assert english.text == english.original and not english.translated
    assert chinese.text == "<zh: 会议定于下周三上午九点举行。>"
    assert chinese.language == "zh"
    assert name.text == "Smith, John"


def test_of_an_email_header_only_the_subject_is_translated():
    email = Section(
        fields=[
            ("From", "علي <a@example.com>"),
            ("Sent", "Tuesday, March 4, 2025 9:12 AM"),
            ("Subject", "جدول الاجتماع"),
        ],
        paragraphs=["مرحبا"],
    )
    document = _translate([email])
    assert [text for _, text in _Fake.calls] == ["جدول الاجتماع", "مرحبا"]
    fields = dict(document.sections[0].fields)
    assert fields["From"].text == "علي <a@example.com>"
    assert fields["Subject"].text == "<ar: جدول الاجتماع>"


def test_a_language_with_no_pack_is_marked_not_sent():
    document = _translate([Section(paragraphs=["안녕하세요 세계"])])
    assert _Fake.calls == []
    (piece,) = document.sections[0].paragraphs
    assert piece.missing == "ko" and not piece.translated
    assert "[Korean, not translated: no language pack] 안녕하세요 세계" in (
        document.render_text()
    )
    assert "No language pack in this copy for Korean" in document.summary()


def test_progress_counts_only_what_is_translated():
    seen = []
    _translate(
        [Section(paragraphs=["English text here.", "中文一", "中文二"])],
        on_progress=lambda done, total, left: seen.append((done, total, left)),
    )
    # The English paragraph costs nothing; no estimate before 5 seconds.
    assert seen == [(1, 2, None), (2, 2, None)]


def test_cancelling_stops_before_the_next_paragraph():
    with pytest.raises(translation.CancelledError):
        _translate([Section(paragraphs=["中文"])], cancelled=lambda: True)


def test_rendered_text_keeps_emails_apart_and_lists_what_was_left_out():
    email = Section(
        fields=[("From", "Ali"), ("To", "Sara"), ("Subject", "جدول")],
        paragraphs=["مرحبا"],
    )
    document = _translate(
        [Section(paragraphs=["Cover page"]), email, email],
        removed=["CUI // Banner", "Page 1 of 2", "CUI // Banner", "Page 2 of 2"],
    )
    text = document.render_text(originals=True)
    assert "──────── Email 1 of 2 ────────" in text
    assert "──────── Email 2 of 2 ────────" in text
    assert "Subject: <ar: جدول>\n    Arabic original: جدول" in text
    assert "<ar: مرحبا>\n    Arabic original: مرحبا" in text
    assert "4 repeated page header, footer and page-number line(s)" in text
    # Each left-out line once, and page numbers not listed at all.
    assert text.endswith(
        "──────── Left out as page headers and footers ────────\nCUI // Banner\n"
    )
    assert document.summary() == (
        "Translated 4 passage(s) from Arabic; left 5 as they were (English, "
        "names, numbers)."
    )
    assert "<ar: مرحبا>\n    Arabic" not in document.render_text()


def test_a_name_in_another_script_is_translated_where_it_stands():
    document = _translate(
        [Section(paragraphs=["Transfer the money to 王建国 tomorrow please"])]
    )
    (piece,) = document.sections[0].paragraphs
    assert _Fake.calls == [("zh", "王建国")]
    assert piece.text == "Transfer the money to [<zh: 王建国>] tomorrow please"
    assert piece.language == "zh" and piece.original.startswith("Transfer the money")


def test_a_run_with_no_pack_marks_the_paragraph_for_a_linguist():
    document = _translate([Section(paragraphs=["Call 김민준 at the office today"])])
    (piece,) = document.sections[0].paragraphs
    assert _Fake.calls == []
    assert piece.missing == "ko" and piece.text == piece.original


_HAS_LANGDETECT = importlib.util.find_spec("langdetect") is not None

SPANISH_CHAT = [
    "nos vemos mañana",
    "dónde estás",
    "llego tarde perdón",
    "te lo mando esta noche",
    "en el mismo lugar",
    "trae los documentos",
    "no le digas a nadie",
    "cuánto cuesta",
    "te llamo luego",
    "estoy en el trabajo",
    "no puedo hablar",
    "abre la puerta",
    "tengo el dinero",
    "mándame la dirección",
    "cambiaron el plan",
    "borra esto después",
    "ya llegó el paquete",
    "pago en efectivo",
]
ENGLISH_CHAT = [
    "OK, see you there.",
    "where are you",
    "running late sorry",
    "I'll send it tonight",
    "same place as before",
    "bring the documents",
    "don't tell anyone",
    "call you back",
    "at work now",
    "open the door",
    "send me the address",
    "they changed the plan",
    "Timestamp",
    "From",
    "Status",
    "Read",
    "John Smith",
    "Maria Lopez",
]


def _flagged(lines, forced=None):
    document = translate_document(
        [Section(paragraphs=list(lines))],
        forced=forced,
        installed={"zh", "ar"},
        translator_for=_Fake,
    )
    return {
        piece.original
        for piece in document.sections[0].paragraphs
        if piece.translated or piece.missing
    }


@pytest.mark.skipif(not _HAS_LANGDETECT, reason="langdetect not installed")
def test_short_spanish_messages_in_a_chat_are_caught_and_english_is_not():
    """Measured with the real detector: a chat of short messages, none long
    enough to be told from English on its own."""
    flagged = _flagged(ENGLISH_CHAT + SPANISH_CHAT)
    caught = flagged & set(SPANISH_CHAT)
    assert len(caught) >= 14  # most of the 18; one-word and unaccented lines slip
    assert flagged <= set(SPANISH_CHAT)  # no English line, name or label


@pytest.mark.skipif(not _HAS_LANGDETECT, reason="langdetect not installed")
def test_an_english_chat_is_not_flagged_even_with_a_language_chosen():
    assert _flagged(ENGLISH_CHAT * 3) == set()
    assert _flagged(ENGLISH_CHAT * 3, forced="es") == set()


@pytest.mark.skipif(not _HAS_LANGDETECT, reason="langdetect not installed")
def test_the_detector_can_weigh_a_few_languages_against_one_another():
    from whispers.translation import detect_language_scores

    (best, _), *_ = detect_language_scores("te llamo luego", ["en", "es"])
    assert best == "es"
    assert {
        code for code, _ in detect_language_scores("te llamo luego", ["en", "es"])
    } <= {
        "en",
        "es",
    }
    # Traditional Chinese has a pack of its own. Asked an open question the
    # detector takes it for Korean, so the two are weighed against each other.
    chinese = ["zh", "zt"]
    traditional = "會議定於下週三舉行，請各部門負責人準時出席。"
    simplified = "会议定于下周三举行，请各部门负责人准时出席。"
    assert detect_language_scores(traditional, chinese)[0][0] == "zt"
    assert detect_language_scores(simplified, chinese)[0][0] == "zh"


@pytest.mark.skipif(not _HAS_LANGDETECT, reason="langdetect not installed")
def test_traditional_chinese_goes_to_its_own_pack_when_this_copy_has_one():
    from whispers.documents import choose_language
    from whispers.translation import detect_language_scores

    traditional = "會議定於下週三舉行，請各部門負責人準時出席。"
    simplified = "会议定于下周三举行，请各部门负责人准时出席。"

    def choose(text, installed):
        return choose_language(
            text, forced=None, installed=installed, detect=detect_language_scores
        )

    assert choose(traditional, {"zh", "zt"}) == ("zt", "")
    assert choose(simplified, {"zh", "zt"}) == ("zh", "")
    # Without the Traditional pack, all Chinese goes to the one there is.
    assert choose(traditional, {"zh"}) == ("zh", "")
