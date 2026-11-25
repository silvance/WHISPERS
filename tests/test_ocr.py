import importlib.util
import os

import pytest

from whispers import ocr
from whispers.ocr import (
    extract_text,
    is_image,
    is_ocr_file,
    is_pdf,
    ocr_available,
    tesseract_lang,
)


@pytest.mark.parametrize(
    "code,expected",
    [
        ("ar", "ara"),
        ("fa", "fas"),
        ("he", "heb"),
        ("zh", "chi_sim"),
        ("ko", "kor"),
        ("en", "eng"),
        ("chi_tra", "chi_tra"),  # already a Tesseract code -> passthrough
        ("xx", "xx"),  # unknown -> passthrough
    ],
)
def test_tesseract_lang_mapping(code, expected):
    assert tesseract_lang(code) == expected


@pytest.mark.parametrize(
    "name,image,pdf,ocrable",
    [
        ("scan.png", True, False, True),
        ("photo.JPG", True, False, True),
        ("doc.pdf", False, True, True),
        ("notes.txt", False, False, False),
        ("clip.mp3", False, False, False),
    ],
)
def test_file_type_predicates(name, image, pdf, ocrable):
    assert is_image(name) is image
    assert is_pdf(name) is pdf
    assert is_ocr_file(name) is ocrable


def test_ocr_available_false_without_binary(monkeypatch):
    monkeypatch.setattr(ocr, "find_tesseract", lambda: None)
    assert ocr_available() is False


def test_ensure_tessdata_prefix_sets_unquoted_path(tmp_path, monkeypatch):
    # Regression: the tessdata dir must reach Tesseract via TESSDATA_PREFIX with no
    # surrounding quotes (the old --tessdata-dir config string embedded quotes in
    # the path, so Tesseract failed to open "<dir>"/eng.traineddata).
    tessdata = tmp_path / "tesseract" / "tessdata"
    tessdata.mkdir(parents=True)
    monkeypatch.setattr(ocr, "bundled_tessdata_dir", lambda: tessdata)
    # Set first, so that what the call under test sets is undone afterwards
    # and does not point later tests' Tesseract at this empty folder.
    monkeypatch.setenv("TESSDATA_PREFIX", "set by the test")
    monkeypatch.delenv("TESSDATA_PREFIX")
    ocr._ensure_tessdata_prefix()
    assert os.environ["TESSDATA_PREFIX"] == str(tessdata)
    assert '"' not in os.environ["TESSDATA_PREFIX"]


def test_extract_text_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "missing.png", lang="eng")


def test_extract_text_rejects_non_ocr_file(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="image or PDF"):
        extract_text(notes, lang="eng")


def test_extract_image_requires_pillow(tmp_path):
    if importlib.util.find_spec("PIL") is not None:
        pytest.skip("Pillow is installed; missing-dependency path not exercised")
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="Pillow is not installed"):
        extract_text(image, lang="eng")


def test_extract_pdf_requires_pypdfium2(tmp_path):
    if importlib.util.find_spec("pypdfium2") is not None:
        pytest.skip("pypdfium2 is installed; missing-dependency path not exercised")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(RuntimeError, match="pypdfium2 is not installed"):
        extract_text(pdf, lang="eng")


# -- Reading scans when the language is left to be detected ------------------


class _Image:
    def __init__(self, name="page"):
        self.name = name

    def crop(self, box):
        return _Image(f"{self.name}-crop")


def _data(lines):
    """image_to_data output for lines of (confidence, words)."""
    out = {k: [] for k in ("text", "conf", "block_num", "par_num", "line_num")}
    out.update(left=[], top=[], width=[], height=[])
    for number, (confidence, words) in enumerate(lines):
        for i, word in enumerate(words.split()):
            out["text"].append(word)
            out["conf"].append(confidence)
            out["block_num"].append(1)
            out["par_num"].append(1)
            out["line_num"].append(number)
            out["left"].append(10 + 50 * i)
            out["top"].append(20 * number)
            out["width"].append(40)
            out["height"].append(15)
    return out


class _Tesseract:
    """Stands in for pytesseract: English reads the page, except two lines of
    Chinese, which only chi_sim reads well."""

    class Output:
        DICT = "dict"

    def __init__(self, script="Latin", chinese=True):
        self.script = script
        self.chinese = chinese
        self.calls = []

    def image_to_osd(self, image, output_type=None):
        return {"script": self.script}

    def image_to_data(self, image, lang, output_type=None):
        self.calls.append(("data", image.name, lang))
        if image.name.endswith("crop"):
            return _data([(92.0 if lang == "chi_sim" else 30.0, "会议 定于")])
        lines = [(95.0, "Meeting notes, part one.")]
        if self.chinese:
            lines.append((14.0, "SWEET PRS LFARER"))
        return _data(lines)

    def image_to_string(self, image, lang):
        self.calls.append(("string", image.name, lang))
        return f" text read with {lang} "


@pytest.fixture
def tesseract(monkeypatch):
    def use(fake, installed=("eng", "ara", "chi_sim", "fra", "osd")):
        monkeypatch.setattr(ocr, "_configure_pytesseract", lambda: fake)
        monkeypatch.setattr(ocr, "_ensure_tessdata_prefix", lambda: None)
        monkeypatch.setattr(ocr, "_installed_tessdata", set(installed))
        return fake

    return use


def test_a_chosen_language_is_read_alongside_english(tesseract):
    fake = tesseract(_Tesseract())
    assert ocr._ocr_image(_Image(), "chi_sim") == "text read with chi_sim+eng"
    assert fake.calls == [("string", "page", "chi_sim+eng")]


def test_lines_english_could_not_read_are_read_in_their_own_script(tesseract):
    fake = tesseract(_Tesseract())
    found = []
    assert ocr._ocr_image(_Image(), None, found) == "text read with chi_sim+eng"
    # Only the non-Latin languages are tried, on the unread lines alone.
    tried = [lang for kind, name, lang in fake.calls if name == "page-crop"]
    assert tried == ["ara", "chi_sim"]
    assert found == ["chi_sim"]

    # The next page tries what the last one found first, and stops there.
    fake.calls = []
    ocr._ocr_image(_Image(), None, found)
    assert [lang for _, name, lang in fake.calls if name == "page-crop"] == ["chi_sim"]


def test_a_page_english_reads_well_is_read_once_in_english(tesseract):
    fake = tesseract(_Tesseract(chinese=False))
    assert ocr._ocr_image(_Image(), None) == "text read with eng"
    assert [kind for kind, _, _ in fake.calls] == ["data", "string"]


def test_a_page_mostly_in_another_script_is_read_in_it(tesseract):
    fake = tesseract(_Tesseract(script="Arabic", chinese=False))
    assert ocr._ocr_image(_Image(), None) == "text read with ara+eng"
    assert fake.calls[-1] == ("string", "page", "ara+eng")


# -- Simplified or Traditional Chinese -----------------------------------------


def test_traditional_chinese_has_its_own_language_data():
    assert tesseract_lang("zt") == "chi_tra"


BOTH = ("eng", "chi_sim", "chi_tra", "osd")


def test_chinese_left_to_be_detected_is_read_with_simplified_data(tesseract):
    """Each kind of Chinese data reads the other's print; nothing measured
    tells the two apart reliably, so detection never picks Traditional."""
    fake = tesseract(_Tesseract(script="Han", chinese=False), installed=BOTH)
    assert ocr._ocr_image(_Image(), None) == "text read with chi_sim+eng"
    assert not any("chi_tra" in lang for _, _, lang in fake.calls)
    # Chosen under From, Traditional print is read with its own data.
    assert ocr._ocr_image(_Image(), "chi_tra") == "text read with chi_tra+eng"


def test_unread_lines_are_tried_in_one_language_for_each_script(tesseract):
    fake = tesseract(
        _Tesseract(), installed=("eng", "ara", "fas", "chi_sim", "chi_tra", "osd")
    )
    ocr._ocr_image(_Image(), None)
    tried = [lang for _, name, lang in fake.calls if name == "page-crop"]
    assert tried == ["ara", "chi_sim"]


def test_lines_are_not_tried_in_another_language_of_the_same_script(tesseract):
    """Arabic data read the page; Farsi data would read the same lines no better."""
    fake = tesseract(
        _Tesseract(script="Arabic", chinese=True),
        installed=("eng", "ara", "fas", "chi_sim", "osd"),
    )
    ocr._ocr_image(_Image(), None)
    tried = [lang for _, name, lang in fake.calls if name == "page-crop"]
    assert tried == ["chi_sim"]


@pytest.mark.skipif(not ocr_available(), reason="Tesseract not installed")
def test_a_clean_page_is_still_read_after_it_is_made_grey():
    """Print is a sliver of a clean page's pixels: stretching the contrast
    with the ends clipped took it for an outlier, and nothing was read."""
    from PIL import Image, ImageDraw, ImageFont

    page = Image.new("RGB", (2400, 1600), "white")
    font = ImageFont.load_default(size=48)
    ImageDraw.Draw(page).text(
        (200, 200), "Meeting notes for the section", "black", font
    )
    assert "Meeting notes" in ocr._ocr_image(page, "eng")
    grey = ocr._for_reading(page)
    assert grey.mode == "L" and grey.getextrema() == (0, 255)


def test_paragraphs_read_with_little_confidence_can_be_left_out(tesseract):
    class Photo(_Tesseract):
        def image_to_data(self, image, lang, output_type=None):
            good = _data([(92.0, "会议 定于")])
            junk = _data([(20.0, "ae SS")])
            for key in good:
                good[key] += junk[key]
            good["par_num"][-2:] = [2, 2]
            return good

    tesseract(Photo(script="Han", chinese=False))
    _, everything = ocr.ocr_paragraphs(_Image(), "chi_sim")
    _, confident = ocr.ocr_paragraphs(_Image(), "chi_sim", min_confidence=45.0)
    assert [text for text, _ in everything] == ["会议定于", "ae SS"]
    assert [text for text, _ in confident] == ["会议定于"]
