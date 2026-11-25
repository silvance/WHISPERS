"""Side-by-side translation of a PDF: what is marked, and what is written."""

import importlib.util

import pytest
from conftest import FRENCH, SPANISH, detect_test_languages

from whispers.passages import read_pdf
from whispers.side_by_side import (
    _latin,
    page_list,
    render_text,
    sha256_of,
    translate_pages,
    write_pdf,
)

NAMES = {"es": "Spanish", "fr": "French"}
SOFTWARE = [("WHISPERS", "test build"), ("Language pack es to English", "1.0")]


class Fake:
    def __init__(self, code):
        self.code = code

    def translate(self, text):
        return f"[{self.code}] the convoy left at six"


def _triage(path):
    pages = read_pdf(path)
    return translate_pages(
        path,
        pages,
        forced=None,
        installed={"es"},
        names=NAMES,
        detect=detect_test_languages,
        translator_for=Fake,
        software=SOFTWARE,
    )


def test_page_numbers_are_listed_in_runs():
    assert page_list([0, 2, 3, 4, 7]) == "1, 3-5, 8"
    assert page_list([]) == ""


def test_text_the_standard_font_cannot_show_is_marked():
    assert _latin("Café “OK” — 5€") == "Café “OK” — 5€"
    assert _latin("Zhang Wei 张伟 said") == "Zhang Wei [?] said"
    assert _latin("Gdańsk") == "Gdansk"


def test_only_foreign_passages_are_marked_and_numbered_per_page(report_pdf):
    triage = _triage(report_pdf)
    assert sorted(triage.marked) == [0, 2]
    (spanish,) = triage.marked[0]
    assert spanish.number == 1
    assert spanish.passage.text == SPANISH
    assert spanish.piece.text == "[es] the convoy left at six"
    (french,) = triage.marked[2]
    assert french.piece.missing == "fr" and not french.piece.translated
    assert triage.sha256 == sha256_of(report_pdf)
    assert triage.summary() == (
        "Foreign-language text on 2 of 3 page(s). Translated 1 passage(s) of "
        "Spanish. No language pack for French: marked, not translated."
    )


def test_the_listing_gives_each_passage_with_where_it_is(report_pdf):
    text = render_text(_triage(report_pdf))
    assert "Unreviewed machine translation" in text
    assert f"SHA-256: {sha256_of(report_pdf)}" in text
    assert "Spanish: 1 passage(s) translated, on page(s) 1" in text
    assert "French: 1 passage(s) NOT translated (no language pack in this copy)" in text
    assert "WHISPERS: test build" in text
    assert f"──────── Page 1 ────────\n\n[1] Spanish\n    Original: {SPANISH}" in text
    assert "    English:  [es] the convoy left at six" in text
    assert f"[1] French: not translated (no language pack)\n    Original: {FRENCH}" in (
        text
    )


def test_the_pdf_has_a_summary_then_each_foreign_page_beside_its_english(
    report_pdf, tmp_path
):
    import pypdfium2 as pdfium

    out = write_pdf(_triage(report_pdf), tmp_path / "report.en.pdf")
    pdf = pdfium.PdfDocument(str(out))
    try:
        assert len(pdf) == 3  # summary, page 1, page 3; page 2 had no foreign text
        texts = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
        width, height = pdf.get_page_size(1)
    finally:
        pdf.close()
    summary, first, third = texts
    assert "Machine translation for language triage" in summary
    assert sha256_of(report_pdf) in summary.replace("\r\n", "")
    assert "Page 1 of 3" in first and "Page 3 of 3" in third
    # The original page is there as it was, its text still searchable.
    assert SPANISH in first and "OK, see you there." in first
    assert "[es] the convoy left at six" in first
    assert "no language pack in this copy, not translated" in third
    assert (width, height) == pytest.approx((1224, 792))


@pytest.mark.skipif(
    importlib.util.find_spec("docx") is None, reason="python-docx not installed"
)
def test_the_word_copy_is_a_table_of_passages(report_pdf, tmp_path):
    import docx

    from whispers.export import side_by_side_docx

    out = side_by_side_docx(_triage(report_pdf), tmp_path / "report.en.docx")
    table = docx.Document(str(out)).tables[0]
    rows = [[cell.text for cell in row.cells] for row in table.rows]
    assert rows[0] == [
        "Page",
        "#",
        "Language",
        "Original",
        "English (machine translation)",
    ]
    assert rows[1] == ["1", "1", "Spanish", SPANISH, "[es] the convoy left at six"]
    assert rows[2] == [
        "3",
        "1",
        "French (no language pack)",
        FRENCH,
        "Not translated: no language pack in this copy.",
    ]


# -- Pages that are turned, or measured from somewhere else --------------------


def _first_page(path):
    (spanish,) = [p for p in read_pdf(path)[0].passages if p.text == SPANISH]
    return read_pdf(path)[0], spanish


def test_passages_are_marked_where_a_turned_page_shows_them(report_pdf, tmp_path):
    """A wide table set sideways: shown turned, read along its lines."""
    import pypdfium2 as pdfium

    from whispers.passages import shown

    _, upright = _first_page(report_pdf)
    pdf = pdfium.PdfDocument(str(report_pdf))
    pdf[0].set_rotation(90)
    turned_path = tmp_path / "turned.pdf"
    pdf.save(str(turned_path))
    pdf.close()

    page, turned = _first_page(turned_path)
    assert (page.width, page.height) == pytest.approx((792, 612))
    assert turned.box == pytest.approx(
        shown(upright.box, (0, 0, 612, 792), 90), abs=0.5
    )
    # The date beside it is still a passage of its own.
    assert "3/12/2023 10:15" in [p.text for p in page.passages]


def test_a_landscape_page_stored_as_a_turned_portrait_one(tmp_path):
    """How report generators often store a landscape page: the content turned
    in the page's own coordinates, and the page turned back for showing."""
    from whispers.side_by_side import _Pdf

    pdf = _Pdf()
    page = pdf.doc.new_page(612, 792)
    for text, x, y in (
        ("3/12/2023 10:15", 60, 500),
        (SPANISH, 200, 500),
        ("Read", 700, 500),
    ):
        # (x, y) on the 792 x 612 page as shown; the text runs up the stored page.
        pdf.text(page, text, 612 - y, x, pdf.regular, 10, turn=(0.0, 1.0, -1.0, 0.0))
    page.set_rotation(90)
    page.gen_content()
    page.close()
    path = tmp_path / "landscape.pdf"
    pdf.save(path)

    shown_page, spanish = _first_page(path)
    assert (shown_page.width, shown_page.height) == pytest.approx((792, 612))
    assert spanish.box[0] == pytest.approx(200, abs=1)
    assert spanish.box[1] <= 500 <= spanish.box[3]
    assert [p.text for p in shown_page.passages] == ["3/12/2023 10:15", SPANISH, "Read"]


def test_passages_are_marked_right_when_the_page_does_not_start_at_its_corner(
    report_pdf, tmp_path
):
    import pypdfium2 as pdfium

    _, before = _first_page(report_pdf)
    pdf = pdfium.PdfDocument(str(report_pdf))
    pdf[0].set_mediabox(40, 60, 652, 852)
    pdf[0].set_cropbox(40, 60, 652, 852)
    moved_path = tmp_path / "moved.pdf"
    pdf.save(str(moved_path))
    pdf.close()

    _, after = _first_page(moved_path)
    expected = (
        before.box[0] - 40,
        before.box[1] - 60,
        before.box[2] - 40,
        before.box[3] - 60,
    )
    assert after.box == pytest.approx(expected, abs=0.5)
    # And the copy is still written, one original page beside its English.
    out = write_pdf(_triage(moved_path), tmp_path / "moved.en.pdf")
    assert pdfium.PdfDocument(str(out)).get_page_size(1) == pytest.approx((1224, 792))


# -- Pictures on a page that has text ------------------------------------------


def _with_pictures(report_pdf, tmp_path):
    """The report with a large picture and a small icon added to its first page."""
    pytest.importorskip("PIL")
    import pypdfium2 as pdfium
    from PIL import Image

    pdf = pdfium.PdfDocument(str(report_pdf))
    page = pdf[0]
    for width, height, x, y in ((240, 180, 300, 300), (24, 24, 60, 600)):
        picture = pdfium.PdfImage.new(pdf)
        picture.set_bitmap(
            pdfium.PdfBitmap.from_pil(Image.new("RGB", (80, 60), "grey"))
        )
        picture.set_matrix(pdfium.PdfMatrix().scale(width, height).translate(x, y))
        page.insert_obj(picture)
    page.gen_content()
    path = tmp_path / "pictures.pdf"
    pdf.save(str(path))
    pdf.close()
    return path


def test_pictures_that_could_hold_text_are_counted_and_said_not_to_be_read(
    report_pdf, tmp_path
):
    path = _with_pictures(report_pdf, tmp_path)
    page = read_pdf(path)[0]
    # The icon is too small to hold text; the picture is found where it is.
    (box,) = page.pictures
    assert box == pytest.approx((300, 300, 540, 480), abs=1)
    assert not page.pictures_read

    triage = _triage(path)
    assert triage.unread_pictures() == (1, [0])
    assert triage.summary().endswith(
        "1 picture(s) on 1 page(s) were not read: any text in them is not covered."
    )
    assert "Pictures NOT read: 1, on page(s) 1." in render_text(triage)


def test_pictures_are_read_when_asked(report_pdf, tmp_path, monkeypatch):
    from whispers import ocr

    path = _with_pictures(report_pdf, tmp_path)
    seen = []

    def read(image, lang, found=None, *, min_confidence=0.0):
        seen.append((image.size, min_confidence))
        return "chi_sim+eng", [("会议定于下周三举行", (30, 60, 330, 120))]

    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_paragraphs", read)
    page = read_pdf(path, pictures=True)[0]
    assert page.pictures_read
    # Only the picture was read, cut out of the page at 300 dots to the inch.
    assert seen == [((1000, 750), 45.0)]
    (chinese,) = [p for p in page.passages if p.script == "han"]
    left, bottom, right, top = chinese.box
    assert 300 <= left < right <= 540 and 300 <= bottom < top <= 480


def test_every_product_says_what_it_cannot_see(report_pdf, tmp_path):
    import pypdfium2 as pdfium

    from whispers.side_by_side import LIMITS

    triage = _triage(report_pdf)
    assert LIMITS in render_text(triage)
    out = write_pdf(triage, tmp_path / "report.en.pdf")
    summary = pdfium.PdfDocument(str(out))[0].get_textpage().get_text_range()
    assert "Finding no foreign-language text does not show" in summary.replace(
        "\r\n", " "
    )
