"""Shared A4 typography and vector-figure placement for the two PDFs."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PAGE_W, PAGE_H = A4
MARGIN = 45
WIDTH = PAGE_W - 2 * MARGIN
INK = "#152C3A"
BODY = "#223642"
TEAL = "#126659"
SOFT = "#EAF3F0"
MUTED = "#49606A"
RULE = "#C6D7D3"
AMBER = "#8A461F"
COVER = "#142F3A"
COVER_LIGHT = "#A8DCCB"
COVER_HEIGHT = 372


def register_fonts(native_name=False):
    kit = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in {
        "Body": "DejaVuSans.ttf", "BodyBold": "DejaVuSans-Bold.ttf",
        "BodyItalic": "DejaVuSans-Oblique.ttf", "Serif": "DejaVuSerif.ttf",
        "SerifBold": "DejaVuSerif-Bold.ttf", "Mono": "DejaVuSansMono.ttf",
    }.items():
        pdfmetrics.registerFont(TTFont(name, str(kit / filename)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic",
                                  boldItalic="BodyBold")
    if native_name:
        font = fitz.Font("china-s")
        pdfmetrics.registerFont(TTFont("StudentCJK", BytesIO(font.buffer)))


def para(text, *, size=9.65, leading=14.5, face="Body", color=BODY, after=9, before=0, align=TA_LEFT):
    return Paragraph(text, ParagraphStyle("text", fontName=face, fontSize=size, leading=leading,
                                           textColor=colors.HexColor(color), spaceAfter=after,
                                           spaceBefore=before, alignment=align, allowWidows=0,
                                           allowOrphans=0, splitLongWords=1))


def section(title):
    return para(title, size=26, leading=32, face="SerifBold", color=INK, after=15)


def heading(title):
    return para(title, size=12, leading=15, face="BodyBold", color=TEAL, after=5, before=6)


def note(content):
    for prefix in ("Figure ", "Table "):
        if content.startswith(prefix):
            label, remainder = content.split(".", 1)
            content = f"<b>{label}.</b>{remainder}"
            break
    return para(content, size=8.15, leading=11.3, color=MUTED, after=8)


def table(headers, entries, widths, numeric=()):
    units = [WIDTH * width for width in widths]
    values = [
        [para(str(item), size=8, leading=10.7, face="BodyBold", color=INK, after=0,
              align=TA_RIGHT if i in numeric else TA_LEFT) for i, item in enumerate(headers)],
        *[[para(str(item), size=8.2, leading=10.9, after=0,
                align=TA_RIGHT if i in numeric else TA_LEFT) for i, item in enumerate(row)] for row in entries],
    ]
    item = Table(values, colWidths=units, hAlign="LEFT", repeatRows=1)
    item.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5.1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.1), ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SOFT)),
        ("LINEABOVE", (0, 0), (-1, 0), .8, colors.HexColor(TEAL)),
        ("LINEBELOW", (0, 0), (-1, 0), .55, colors.HexColor(RULE)),
        ("LINEBELOW", (0, 1), (-1, -1), .32, colors.HexColor(RULE)),
        ("LINEBELOW", (0, -1), (-1, -1), .55, colors.HexColor(RULE)),
    ]))
    return item


def code(lines):
    commands = para("<br/>".join(lines), size=8.2, leading=13.5, face="Mono", color=INK, after=0)
    item = Table([[commands]], colWidths=[WIDTH], hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(SOFT)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return item


class CoverTrace(Flowable):
    def __init__(self, times, amplitudes, record):
        super().__init__()
        self.width = WIDTH
        self.height = 90
        self.times = times
        self.amplitudes = amplitudes
        self.record = record

    def draw(self):
        canvas = self.canv
        x0, x1 = self.times[0], self.times[-1]
        lo, hi = min(self.amplitudes), max(self.amplitudes)
        canvas.setStrokeColor(colors.HexColor("#31505A"))
        canvas.setLineWidth(.45)
        for x in range(17):
            offset = x * self.width / 16
            canvas.line(offset, 22, offset, 83)
        for y in (22, 42, 62, 83):
            canvas.line(0, y, self.width, y)
        curve = canvas.beginPath()
        for index, (t, mv) in enumerate(zip(self.times, self.amplitudes)):
            x = self.width * (t - x0) / (x1 - x0)
            y = 27 + 51 * (mv - lo) / max(hi - lo, 1e-6)
            if index == 0:
                curve.moveTo(x, y)
            else:
                curve.lineTo(x, y)
        canvas.setStrokeColor(colors.HexColor(COVER_LIGHT))
        canvas.setLineWidth(1.25)
        canvas.drawPath(curve)
        canvas.setFillColor(colors.HexColor(COVER_LIGHT))
        canvas.setFont("Body", 7.5)
        canvas.drawString(0, 5, f"PhysioNet MIT-BIH  •  Record {self.record}, MLII")
        canvas.drawRightString(self.width, 5, f"{x0:.2f} to {x1:.2f} s")


class VectorFigure(Flowable):
    def __init__(self, source, placements):
        super().__init__()
        self.source = source
        self.placements = placements
        self.width = WIDTH
        with fitz.open(source) as doc:
            rect = doc[0].rect
        self.height = WIDTH * rect.height / rect.width

    def draw(self):
        x, y = self.canv.absolutePosition(0, 0)
        self.placements.append((self.canv.getPageNumber(), fitz.Rect(
            x, PAGE_H - y - self.height, x + self.width, PAGE_H - y,
        ), self.source))


def build_pdf(path, pages, title, short_title, author, draft, placements, *, cover_band=False):
    def decorate(canvas, doc):
        canvas.setTitle(title)
        canvas.setAuthor(author)
        canvas.setSubject("Biomedical Signal Processing | MIT-BIH Arrhythmia Database")
        is_cover = cover_band and doc.page == 1
        if is_cover:
            canvas.setFillColor(colors.HexColor(COVER))
            canvas.rect(0, PAGE_H - COVER_HEIGHT, PAGE_W, COVER_HEIGHT, stroke=0, fill=1)
        canvas.setStrokeColor(colors.HexColor(RULE))
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.setLineWidth(.65)
        canvas.line(MARGIN, 32, PAGE_W - MARGIN, 32)
        canvas.setFont("Body", 7.5)
        canvas.drawString(MARGIN, 19, short_title)
        canvas.drawRightString(PAGE_W - MARGIN, 19, f"{doc.page:02d}  /  {len(pages):02d}")
        canvas.setFont("BodyBold", 7.8)
        canvas.setFillColor(colors.HexColor(COVER_LIGHT if is_cover else TEAL))
        canvas.drawString(MARGIN, PAGE_H - 31, "Biomedical Signal Processing")
        if draft:
            canvas.setFillColor(colors.HexColor("#F3CB97" if is_cover else AMBER))
            canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 31, "DRAFT FOR STUDENT REVIEW")
        if not is_cover:
            canvas.line(MARGIN, PAGE_H - 39, PAGE_W - MARGIN, PAGE_H - 39)

    with TemporaryDirectory(prefix="pdf-", dir=path.parent) as directory:
        intermediate = Path(directory) / "layout.pdf"
        builder = SimpleDocTemplate(str(intermediate), pagesize=A4, leftMargin=MARGIN - 6,
                                    rightMargin=MARGIN - 6, topMargin=55, bottomMargin=43)
        story = []
        for i, page in enumerate(pages):
            if i:
                story.append(PageBreak())
            story.extend(page)
        builder.build(story, onFirstPage=decorate, onLaterPages=decorate)
        with fitz.open(intermediate) as pdf:
            if len(pdf) != len(pages):
                raise ValueError(f"{path.name} overflowed: expected {len(pages)} pages, got {len(pdf)}")
            for number, rectangle, asset in placements:
                with fitz.open(asset) as graphic:
                    pdf[number - 1].show_pdf_page(rectangle, graphic, 0)
            rendered = Path(directory) / path.name
            pdf.save(rendered, garbage=4, deflate=True)
        rendered.replace(path)
