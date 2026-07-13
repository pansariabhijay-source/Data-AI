"""Generate Presentation.pdf — a short, business-led spoken script for a director briefing."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ── palette ──────────────────────────────────────────────────────────────────
INK      = colors.HexColor("#0F172A")   # slate-900, body text
MUTED    = colors.HexColor("#475569")   # slate-600, cues
ACCENT   = colors.HexColor("#4F46E5")   # indigo-600, headings
ACCENT2  = colors.HexColor("#06B6D4")   # cyan-500, rule
CARD_BG  = colors.HexColor("#EEF2FF")   # indigo-50, callout
CARD_LN  = colors.HexColor("#C7D2FE")   # indigo-200
FAINT    = colors.HexColor("#94A3B8")

OUT = "/Users/abhijaypansari/Documents/Data-AI/Presentation.pdf"

styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, parent=styles["Normal"], **kw)

title_style   = S("Title2", fontName="Helvetica-Bold", fontSize=26, leading=30,
                   textColor=INK, spaceAfter=2)
subtitle_style= S("Sub", fontName="Helvetica", fontSize=11.5, leading=15,
                   textColor=MUTED, spaceAfter=2)
kicker_style  = S("Kick", fontName="Helvetica-Bold", fontSize=9.5, leading=12,
                   textColor=ACCENT2, spaceAfter=4)
head_style    = S("Head", fontName="Helvetica-Bold", fontSize=13.5, leading=16,
                   textColor=ACCENT, spaceBefore=12, spaceAfter=4)
body_style    = S("Body", fontName="Helvetica", fontSize=11, leading=16.5,
                   textColor=INK, spaceAfter=7, alignment=TA_LEFT)
cue_style     = S("Cue", fontName="Helvetica-Oblique", fontSize=9.5, leading=13,
                   textColor=MUTED, spaceAfter=7)
card_h_style  = S("CardH", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                   textColor=ACCENT, spaceAfter=3)
card_b_style  = S("CardB", fontName="Helvetica", fontSize=10, leading=14.5,
                   textColor=INK)
foot_style    = S("Foot", fontName="Helvetica", fontSize=8, leading=10,
                   textColor=FAINT, alignment=TA_CENTER)

story = []

# ── header ───────────────────────────────────────────────────────────────────
story += [
    Paragraph("SPOKEN SCRIPT · DIRECTOR BRIEFING", kicker_style),
    Paragraph("Axiom", title_style),
    Paragraph("The automatic data scientist &mdash; a five-minute walkthrough to read aloud.",
              subtitle_style),
    Spacer(1, 8),
    HRFlowable(width="100%", thickness=2, color=ACCENT2, spaceAfter=10),
]

# ── delivery callout ─────────────────────────────────────────────────────────
tips = Paragraph(
    "<b>How to use this:</b> Read it aloud, roughly word-for-word &mdash; it runs about "
    "five minutes at a calm pace. <i>Italic lines</i> are delivery cues, not spoken. "
    "Slow down, pause where marked, and look up on the bold phrases.",
    card_b_style)
card = Table([[tips]], colWidths=[6.6 * inch])
card.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
    ("BOX", (0, 0), (-1, -1), 0.75, CARD_LN),
    ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ("TOPPADDING", (0, 0), (-1, -1), 9),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
]))
story += [card, Spacer(1, 6)]


def section(title, seconds, paras):
    story.append(Paragraph(f"{title}  &nbsp;<font size=8 color='#94A3B8'>~{seconds}</font>",
                           head_style))
    for p in paras:
        if isinstance(p, tuple) and p[0] == "cue":
            story.append(Paragraph(p[1], cue_style))
        else:
            story.append(Paragraph(p, body_style))


# ── the script ───────────────────────────────────────────────────────────────
section("1 &nbsp;·&nbsp; The opening", "30s", [
    ("cue", "Calm, unhurried. Make eye contact before you start."),
    "&ldquo;Thanks for making the time. In the next five minutes I want to show you "
    "something we&rsquo;ve built called <b>Axiom</b> &mdash; and why I think it matters for us.",
    "Let me start with a simple question: how long does it take us today to turn a "
    "spreadsheet of data into a decision we can trust? Usually the answer is weeks, and it "
    "needs a specialist. <b>Axiom does it in minutes.</b>&rdquo;",
])

section("2 &nbsp;·&nbsp; The problem", "45s", [
    "&ldquo;Here&rsquo;s the problem. We&rsquo;re sitting on a lot of data &mdash; "
    "transactions, customers, operations. Inside that data are answers: which transactions "
    "are likely fraud, which customers are about to leave, what&rsquo;s driving our costs.",
    "But getting those answers is hard. It takes a trained data scientist to clean the "
    "data, build the model, test it properly, and write it up. That&rsquo;s expensive, "
    "it&rsquo;s slow, and &mdash; the part people underestimate &mdash; it&rsquo;s easy to "
    "get subtly wrong. A model can look brilliant in testing and quietly fail in the real "
    "world. By then it&rsquo;s already cost us.&rdquo;",
])

section("3 &nbsp;·&nbsp; The solution", "45s", [
    "&ldquo;Axiom solves that. Think of it as an <b>automatic data scientist</b> that "
    "anyone on the team can use.",
    "You upload a spreadsheet. You point at the one column you want to predict &mdash; say, "
    "&lsquo;is this transaction fraud?&rsquo;. You click run. That&rsquo;s it.",
    ("cue", "Slow down here &mdash; this is the core idea."),
    "Behind the scenes it does everything an expert would: it cleans the data, builds the "
    "right signals, trains and compares several models, picks the best one, checks it for "
    "mistakes, and hands you a clear report &mdash; in plain language, in a few minutes. "
    "<b>No coding, no specialist required.</b>&rdquo;",
])

section("4 &nbsp;·&nbsp; How it works, in one breath", "30s", [
    "&ldquo;Under the hood it&rsquo;s a pipeline of steps, each doing a specific job and "
    "handing off to the next &mdash; collect, clean, build features, train, check for "
    "errors, improve, and finalise. You don&rsquo;t need to know any of that to use it. "
    "You just see the result: a working model and an honest report.&rdquo;",
])

section("5 &nbsp;·&nbsp; Why it matters to the business", "60s", [
    "&ldquo;So why does this matter to us? Three things.",
    "<b>Speed.</b> Work that took weeks now takes minutes. We can test an idea in an "
    "afternoon instead of committing a team for a month.",
    "<b>Cost.</b> It runs on our own machines. There&rsquo;s no per-use AI bill, and it "
    "doesn&rsquo;t need us to hire a specialist for every new question &mdash; the "
    "expertise is built in.",
    "<b>Trust and privacy.</b> The data never leaves our environment; nothing is sent to "
    "an outside AI service. And Axiom is deliberately honest &mdash; instead of just "
    "claiming a model is &lsquo;accurate&rsquo;, it tells you, in real terms, how many "
    "frauds you&rsquo;d actually catch for a given amount of review effort. That&rsquo;s "
    "the number a business can plan around.&rdquo;",
])

section("6 &nbsp;·&nbsp; Where it shines: fraud", "30s", [
    "&ldquo;We tuned it especially for <b>fraud detection</b>, because that&rsquo;s where "
    "the money is and where the mistakes are most costly. Fraud is rare and it changes over "
    "time, so it&rsquo;s easy to build a model that looks good on paper and misses the real "
    "thing. Axiom handles those traps properly, and reports results the way a fraud team "
    "actually needs to see them.&rdquo;",
])

section("7 &nbsp;·&nbsp; The close and the ask", "40s", [
    "&ldquo;The one line I&rsquo;d leave you with: it&rsquo;s fast, it&rsquo;s private, it "
    "costs almost nothing to run, and it&rsquo;s honest about what it can and can&rsquo;t "
    "do. That combination is rare.",
    ("cue", "Land this next part directly &mdash; it&rsquo;s the ask."),
    "So my ask is simple: let me run Axiom on one of our real datasets &mdash; you pick the "
    "problem you care about &mdash; and I&rsquo;ll show you a working model and a report by "
    "the end of the week. If it delivers, we&rsquo;ve made data science something the whole "
    "team can use, not just a specialist.",
    "Thank you &mdash; happy to take any questions.&rdquo;",
])

# ── anticipated questions ────────────────────────────────────────────────────
story += [Spacer(1, 6),
          HRFlowable(width="100%", thickness=1, color=CARD_LN, spaceAfter=8)]
story.append(Paragraph("If they ask &mdash; quick answers", head_style))

qa = [
    ("&ldquo;Does this use ChatGPT / send our data anywhere?&rdquo;",
     "No. The thinking is classic, proven maths running on our own machines. Nothing "
     "leaves our environment, and there&rsquo;s no per-run AI cost."),
    ("&ldquo;Does it replace our analysts?&rdquo;",
     "No &mdash; it removes the slow, repetitive setup so they spend time on judgement and "
     "action, and it lets non-specialists get a first answer on their own."),
    ("&ldquo;How do we know we can trust the model?&rdquo;",
     "It reports honest, real-world numbers &mdash; how many frauds you catch for a given "
     "review effort &mdash; and it&rsquo;s built to avoid the mistakes that make models "
     "look good in testing but fail live."),
]
rows = []
for q, a in qa:
    rows.append([Paragraph(q, card_h_style)])
    rows.append([Paragraph(a, card_b_style)])
qtable = Table(rows, colWidths=[6.6 * inch])
qtable.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
    ("BOX", (0, 0), (-1, -1), 0.75, CARD_LN),
    ("LINEBELOW", (0, 1), (-1, 1), 0.5, CARD_LN),
    ("LINEBELOW", (0, 3), (-1, 3), 0.5, CARD_LN),
    ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(qtable)


# ── footer on every page ─────────────────────────────────────────────────────
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(FAINT)
    canvas.drawCentredString(LETTER[0] / 2, 0.45 * inch,
                             "Axiom  ·  Presentation script  ·  ~5 minutes")
    canvas.restoreState()


doc = SimpleDocTemplate(
    OUT, pagesize=LETTER,
    leftMargin=0.95 * inch, rightMargin=0.95 * inch,
    topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    title="Axiom — Presentation Script", author="Axiom",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
