"""Shared A4 typography and vector-figure placement for the two PDFs."""

from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PAGE_W, PAGE_H = A4
MARGIN = 45
WIDTH = PAGE_W - 2 * MARGIN
INK = "#172F43"
BODY = "#223B4E"
TEAL = "#10677A"
SOFT = "#EDF5F7"
MUTED = "#456173"
RULE = "#C8D8DE"
AMBER = "#8A461F"


def register_fonts():
    kit = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in {
        "Body": "DejaVuSans.ttf", "BodyBold": "DejaVuSans-Bold.ttf",
        "BodyItalic": "DejaVuSans-Oblique.ttf", "Serif": "DejaVuSerif.ttf",
        "SerifBold": "DejaVuSerif-Bold.ttf", "Mono": "DejaVuSansMono.ttf",
    }.items():
        pdfmetrics.registerFont(TTFont(name, str(kit / filename)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic",
                                  boldItalic="BodyBold")


def para(text, *, size=9.6, leading=14.4, face="Body", color=BODY, after=8, before=0):
    return Paragraph(text, ParagraphStyle("text", fontName=face, fontSize=size, leading=leading,
                                           textColor=colors.HexColor(color), spaceAfter=after,
                                           spaceBefore=before, alignment=TA_LEFT, allowWidows=0,
                                           allowOrphans=0, splitLongWords=1))


def section(title):
    return para(title, size=25, leading=30, face="SerifBold", color=INK, after=14)


def heading(title):
    return para(title, size=12, leading=15, face="BodyBold", color=TEAL, after=5, before=6)


def note(content):
    return para(content, size=8.15, leading=11.3, color=MUTED, after=8)


def rule():
    line = Table([[""]], colWidths=[WIDTH], rowHeights=[1])
    line.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(RULE))]))
    return line


def table(headers, entries, widths):
    units = [WIDTH * width for width in widths]
    values = [
        [para(str(item), size=8, leading=10.7, face="BodyBold", color="#FFFFFF", after=0) for item in headers],
        *[[para(str(item), size=8.2, leading=10.9, after=0) for item in row] for row in entries],
    ]
    item = Table(values, colWidths=units, hAlign="LEFT", repeatRows=1)
    item.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5.1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.1), ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(SOFT), colors.white]),
        ("LINEBELOW", (0, -1), (-1, -1), .55, colors.HexColor(RULE)),
    ]))
    return item


def callout(label, message):
    item = Table([[para(label, size=8, leading=11, face="BodyBold", color=TEAL, after=4),
                   ], [para(message, size=10.1, leading=15, face="Body", color=INK, after=0)]],
                 colWidths=[WIDTH], hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(SOFT)),
        ("LINEBEFORE", (0, 0), (0, -1), 2, colors.HexColor(TEAL)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 9), ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 0), ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
    ]))
    return item


def code(lines):
    return callout("TERMINAL", "<br/>".join(
        f'<font name="Mono" size="8.2">{line}</font>' for line in lines
    ))


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


def build_pdf(path, pages, title, short_title, author, draft, placements):
    def decorate(canvas, doc):
        canvas.setTitle(title)
        canvas.setAuthor(author)
        canvas.setSubject("Biomedical Signal Processing | MIT-BIH Arrhythmia Database")
        canvas.setStrokeColor(colors.HexColor(RULE))
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.setLineWidth(.65)
        canvas.line(MARGIN, 32, PAGE_W - MARGIN, 32)
        canvas.setFont("Body", 7.5)
        canvas.drawString(MARGIN, 19, short_title)
        canvas.drawRightString(PAGE_W - MARGIN, 19, f"{doc.page:02d}  /  {len(pages):02d}")
        canvas.setFont("BodyBold", 7.8)
        canvas.setFillColor(colors.HexColor(TEAL))
        canvas.drawString(MARGIN, PAGE_H - 31, "BIOMEDICAL SIGNAL PROCESSING")
        if draft:
            canvas.setFillColor(colors.HexColor(AMBER))
            canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 31, "DRAFT FOR STUDENT REVIEW")
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
