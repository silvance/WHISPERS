"""The passages of text on each page of a PDF, and where each one sits.

Forensic reports (Cellebrite, Magnet AXIOM) lay messages out in tables,
label/value blocks and chat bubbles, and every tool and version does it
differently. Rather than parse each layout, text is grouped by where it sits
on the page:

* each line is cut where a wide gap separates columns, so a table row
  becomes its cells (number, time, sender, message, status);
* a cell that wraps onto the next lines is put back together, because its
  lines sit directly under one another, aligned, in the same script;
* a scanned page is read with OCR, which reports paragraphs and their
  boxes the same way.

Each passage keeps its box in PDF points, so a side-by-side copy can mark it
on the original page. Boxes are in the page **as it is shown**: a landscape
page is often stored as a portrait one with a rotation, and a page's own
coordinates need not start at its corner, so both are undone first (see
:func:`shown`). Nothing here translates or decides what is foreign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .documents import clean, dominant_script, join_wrapped
from .transcription import CancelCallback, CancelledError, ProgressCallback

PathLike = Union[str, Path]
# (left, bottom, right, top) in PDF points, origin at the bottom left.
Box = Tuple[float, float, float, float]

# Two characters on a line further apart than this many times the font size
# are in different columns. A word space is about a quarter of it.
_COLUMN_GAP = 0.8
# A line no further below the one above it than this many times the font
# size carries on the same passage; table rows and chat bubbles sit further
# apart than lines of one paragraph.
_LINE_GAP = 0.8
# How far, in font sizes, the edges of two lines of one passage may differ.
_ALIGNED = 1.5
# A picture smaller than this on either side, in points (72 to the inch), is
# an icon, an avatar or a logo: nothing that could hold text worth reading.
_PICTURE_MIN_SIDE = 72.0
# A picture covering this share of a page that has text of its own is a scan
# with a text layer: its text has already been read.
_SCAN_SHARE = 0.85
# Words OCR finds in a photograph of something other than text are read with
# little confidence (0-100); a paragraph below this is left out.
_PICTURE_MIN_CONFIDENCE = 45.0


@dataclass
class Passage:
    """A run of text on a page: a table cell, a paragraph, a message."""

    page: int
    text: str
    box: Box
    # Dominant script (see whispers.documents.dominant_script); None for numbers
    # and punctuation.
    script: Optional[str]
    size: float = 10.0


@dataclass
class PageText:
    """One page: its size, its text as extracted, and its passages."""

    index: int
    width: float
    height: float
    text: str
    passages: List[Passage] = field(default_factory=list)
    # The Tesseract languages a scanned page was read with; None when the
    # page had text of its own.
    ocr: Optional[str] = None
    # Pictures on a page that has text of its own, where they are shown. Any
    # text in them is in the passages only if ``pictures_read``.
    pictures: List[Box] = field(default_factory=list)
    pictures_read: bool = False


@dataclass
class _Segment:
    text: str
    box: Box
    size: float

    @property
    def script(self) -> Optional[str]:
        return dominant_script(self.text)


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def shown(box: Box, page_box: Box, rotation: int) -> Box:
    """``box``, given in a page's own coordinates, as the page is shown.

    ``page_box`` is the part of the page that is shown and ``rotation`` how
    far it is turned clockwise for showing (0, 90, 180 or 270). The result is
    measured from the bottom left of the page as a reader sees it.
    """
    left, bottom, right, top = page_box
    width, height = right - left, top - bottom
    x0, y0, x1, y1 = box[0] - left, box[1] - bottom, box[2] - left, box[3] - bottom
    if rotation % 360 == 90:
        return (y0, width - x1, y1, width - x0)
    if rotation % 360 == 180:
        return (width - x1, height - y1, width - x0, height - y0)
    if rotation % 360 == 270:
        return (height - y1, x0, height - y0, x1)
    return (x0, y0, x1, y1)


def _text_turn(chars: Sequence[Tuple[str, Optional[Box], float]]) -> int:
    """How the text on a page runs, as shown: 0 across, 90 up or 270 down it.

    A wide table is sometimes set sideways on an upright page, to be read
    with the head turned. Most pairs of neighbouring characters then sit one
    above the other instead of side by side.
    """
    up = down = across = 0
    last: Optional[Box] = None
    for _, box, _ in chars:
        if box is None:
            last = None
            continue
        if last is not None:
            dx = (box[0] + box[2]) - (last[0] + last[2])
            dy = (box[1] + box[3]) - (last[1] + last[3])
            if abs(dy) <= abs(dx):
                across += 1
            elif dy > 0:
                up += 1
            else:
                down += 1
        last = box
    if max(up, down) <= across:
        return 0
    return 90 if up > down else 270


def _turned(box: Box, turn: int, back: bool = False) -> Box:
    """``box`` with the page turned so that its text runs across, or back."""
    x0, y0, x1, y1 = box
    if (turn == 90) != back:
        return (y0, -x1, y1, -x0)
    return (-y1, x0, -y0, x1)


def _page_passages(page, index: int) -> List[Passage]:
    """The passages of a page that has text of its own."""
    chars = _page_chars(page)
    turn = _text_turn(chars)
    if turn:
        chars = [
            (ch, None if box is None else _turned(box, turn), size)
            for ch, box, size in chars
        ]
    passages = group(_segments(chars), index)
    if turn:
        for passage in passages:
            passage.box = _turned(passage.box, turn, back=True)
    return passages


def _page_box(page) -> Box:
    """The part of ``page`` that is shown: its crop box within its media box."""
    get = getattr(page, "get_bbox", None)
    if get is not None:
        left, bottom, right, top = get()
        return (left, bottom, right, top)
    media, crop = page.get_mediabox(), page.get_cropbox()
    return (
        max(media[0], crop[0]),
        max(media[1], crop[1]),
        min(media[2], crop[2]),
        min(media[3], crop[3]),
    )


def _segments(
    chars: Iterable[Tuple[str, Optional[Box], float]],
) -> List[_Segment]:
    """Lines of characters cut into column segments.

    ``chars`` is the page's characters in pdfium's reading order, each with
    its box and font size; a box of None is a space or line break pdfium
    put in. pdfium orders a right-to-left line from the right, so a gap is
    measured whichever way the line runs.
    """
    out: List[_Segment] = []
    text = ""
    box: Optional[Box] = None
    size = 0.0
    last: Optional[Box] = None

    def close() -> None:
        nonlocal text, box, size, last
        stripped = clean(text).strip()
        if stripped and box is not None:
            out.append(_Segment(stripped, box, size or 10.0))
        text, box, size, last = "", None, 0.0, None

    for ch, char_box, char_size in chars:
        if ch in "\r\n":
            close()
            continue
        if char_box is None or ch.isspace():
            if text and not text.endswith(" "):
                text += " "
            continue
        if last is not None:
            gap = max(char_box[0] - last[2], last[0] - char_box[2])
            if gap > _COLUMN_GAP * max(char_size, size, 4.0):
                close()
        text += ch
        box = char_box if box is None else _union(box, char_box)
        size = max(size, char_size)
        last = char_box
    close()
    return out


@dataclass
class _Block:
    segments: List[_Segment]
    box: Box
    script: Optional[str]

    @property
    def size(self) -> float:
        return max(s.size for s in self.segments)

    def continues(self, segment: _Segment) -> bool:
        """Whether ``segment`` is the next line of this passage."""
        script = segment.script
        if script is not None and script != self.script:
            return False
        if self.script is None:
            # Numbers and dates do not gather text under them.
            return False
        above = self.segments[-1].box
        size = max(self.size, segment.size)
        gap = above[1] - segment.box[3]
        if not -0.3 * size <= gap <= _LINE_GAP * size:
            return False
        left, _, right, _ = self.box
        if segment.box[0] > right or segment.box[2] < left:
            return False
        return (
            abs(segment.box[0] - left) <= _ALIGNED * size
            or abs(segment.box[2] - right) <= _ALIGNED * size
        )


def group(segments: Sequence[_Segment], page: int) -> List[Passage]:
    """Segments gathered into passages, in reading order (top down, then across)."""
    blocks: List[_Block] = []
    for segment in segments:
        # Only the last few passages are open: a cell's next line comes soon
        # after its first in pdfium's order.
        for block in reversed(blocks[-12:]):
            if block.continues(segment):
                block.segments.append(segment)
                block.box = _union(block.box, segment.box)
                break
        else:
            blocks.append(_Block([segment], segment.box, segment.script))
    passages = []
    for block in blocks:
        text = ""
        for segment in block.segments:
            text = join_wrapped(text, segment.text) if text else segment.text
        passages.append(
            Passage(page, text, block.box, dominant_script(text), block.size)
        )
    passages.sort(key=lambda p: (-round(p.box[3]), p.box[0]))
    return passages


def _page_chars(page) -> List[Tuple[str, Optional[Box], float]]:
    """The characters of ``page`` with their boxes as the page is shown."""
    import pypdfium2.raw as raw

    page_box, rotation = _page_box(page), page.get_rotation()
    textpage = page.get_textpage()
    try:
        handle = textpage.raw
        chars: List[Tuple[str, Optional[Box], float]] = []
        for i in range(textpage.count_chars()):
            ch = chr(raw.FPDFText_GetUnicode(handle, i))
            if ch in "\r\n" or raw.FPDFText_IsGenerated(handle, i) == 1:
                chars.append((ch, None, 0.0))
                continue
            box = shown(textpage.get_charbox(i, loose=True), page_box, rotation)
            chars.append((ch, box, raw.FPDFText_GetFontSize(handle, i)))
        return chars
    finally:
        textpage.close()


def _pictures(page, width: float, height: float) -> List[Box]:
    """Where the pictures big enough to hold text are, as the page is shown."""
    import pypdfium2.raw as raw

    page_box, rotation = _page_box(page), page.get_rotation()
    boxes: List[Box] = []
    for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=4):
        bounds = getattr(obj, "get_bounds", None) or obj.get_pos
        try:
            left, bottom, right, top = shown(bounds(), page_box, rotation)
        except Exception:  # noqa: BLE001 - a picture pdfium cannot place
            continue
        left, bottom = max(0.0, left), max(0.0, bottom)
        right, top = min(width, right), min(height, top)
        if min(right - left, top - bottom) < _PICTURE_MIN_SIDE:
            continue
        if (right - left) * (top - bottom) >= _SCAN_SHARE * width * height:
            continue
        boxes.append((left, bottom, right, top))
    return boxes


def _page_text(page) -> str:
    textpage = page.get_textpage()
    try:
        return clean(textpage.get_text_range()).strip()
    finally:
        textpage.close()


def ocr_passages(
    paragraphs: Sequence[Tuple[str, Tuple[int, int, int, int]]],
    page: int,
    *,
    height: float,
    dpi: float,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> List[Passage]:
    """OCR paragraphs (pixel boxes from the top) as passages in PDF points.

    ``origin`` is where the image that was read starts on the page, in pixels
    from the page's top left: (0, 0) for a whole page, the corner of the
    picture for a picture cut out of one.
    """
    scale = 72.0 / dpi
    out = []
    for text, (left, top, right, bottom) in paragraphs:
        box = (
            (left + origin[0]) * scale,
            height - (bottom + origin[1]) * scale,
            (right + origin[0]) * scale,
            height - (top + origin[1]) * scale,
        )
        out.append(Passage(page, text, box, dominant_script(text), 10.0))
    out.sort(key=lambda p: (-round(p.box[3]), p.box[0]))
    return out


def _read_pictures(
    page, boxes: Sequence[Box], index: int, height: float, lang, found
) -> List[Passage]:
    """The text OCR finds in the pictures at ``boxes`` on ``page``."""
    from . import ocr

    dpi = ocr._PDF_RENDER_DPI
    scale = dpi / 72.0
    image = page.render(scale=scale).to_pil()
    passages: List[Passage] = []
    try:
        for left, bottom, right, top in boxes:
            origin = (int(left * scale), int((height - top) * scale))
            crop = image.crop(
                (
                    origin[0],
                    origin[1],
                    int(right * scale),
                    int((height - bottom) * scale),
                )
            )
            try:
                _, paragraphs = ocr.ocr_paragraphs(
                    crop, lang, found, min_confidence=_PICTURE_MIN_CONFIDENCE
                )
            finally:
                crop.close()
            passages += ocr_passages(
                paragraphs, index, height=height, dpi=dpi, origin=origin
            )
    finally:
        image.close()
    return passages


def read_pdf(
    path: PathLike,
    *,
    lang: Optional[str] = None,
    pictures: bool = False,
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCallback] = None,
) -> List[PageText]:
    """Every page of a PDF with its passages; scanned pages are read with OCR.

    ``lang`` is a Tesseract language for scanned pages (read alongside
    English), or None to detect each page's script. A page with text of its
    own can also hold pictures with text in them (a screenshot of a chat, a
    photographed letter): they are always counted, and with ``pictures`` they
    are read with OCR too, which takes a few seconds for each.
    """
    from . import ocr

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "pypdfium2 is not installed. Install the OCR extra with:  "
            "pip install 'whispers[ocr]'"
        ) from exc
    pdf = pdfium.PdfDocument(str(path))
    found: List[str] = []
    pages: List[PageText] = []
    try:
        total = len(pdf)
        for index in range(total):
            if cancelled is not None and cancelled():
                raise CancelledError("Cancelled.")
            page = pdf[index]
            try:
                width, height = page.get_size()
                text = _page_text(page)
                if len(text) >= ocr._MIN_EMBEDDED_CHARS:
                    if progress is not None:
                        progress(f"Reading page {index + 1} of {total}…")
                    passages = _page_passages(page, index)
                    boxes = _pictures(page, width, height)
                    read = bool(boxes) and pictures and ocr.ocr_available()
                    if read:
                        if progress is not None:
                            progress(
                                f"Reading {len(boxes)} picture(s) on page "
                                f"{index + 1} of {total}…"
                            )
                        passages += _read_pictures(
                            page, boxes, index, height, lang, found
                        )
                        passages.sort(key=lambda p: (-round(p.box[3]), p.box[0]))
                    pages.append(
                        PageText(
                            index,
                            width,
                            height,
                            text,
                            passages,
                            pictures=boxes,
                            pictures_read=read,
                        )
                    )
                    continue
                if not ocr.ocr_available():
                    pages.append(PageText(index, width, height, text))
                    continue
                if progress is not None:
                    progress(f"Reading page {index + 1} of {total} (scanned)…")
                dpi = ocr._PDF_RENDER_DPI
                image = page.render(scale=dpi / 72).to_pil()
                try:
                    used, paragraphs = ocr.ocr_paragraphs(image, lang, found)
                finally:
                    image.close()
                passages = ocr_passages(paragraphs, index, height=height, dpi=dpi)
                pages.append(
                    PageText(
                        index,
                        width,
                        height,
                        "\n\n".join(p.text for p in passages),
                        passages,
                        ocr=used,
                    )
                )
            finally:
                page.close()
        return pages
    finally:
        pdf.close()
