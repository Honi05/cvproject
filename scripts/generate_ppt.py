"""Generate the Sapienza-templated presentation (CNN vs YOLO), 7-min talk."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches as I, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figs"
OUT = ROOT / "outputs" / "Chess_CNN_vs_YOLO_Sapienza.pptx"

MAROON = RGBColor(0x82, 0x24, 0x33)
TEAL = RGBColor(0x15, 0x70, 0x7E)
DARK = RGBColor(0x26, 0x26, 0x29)
BODY = RGBColor(0x59, 0x59, 0x5C)
GREY = RGBColor(0x8A, 0x8A, 0x8C)
BG = RGBColor(0xEA, 0xEE, 0xEF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HFONT = "Lato"
BFONT = "Lato"

AUTHOR = "Honi Arora"
COURSE = "Computer Vision & Deep Learning"
FOOTER = "Chess: Custom CNN vs YOLO — Honi Arora"

prs = Presentation()
prs.slide_width = I(13.333)
prs.slide_height = I(7.5)
BLANK = prs.slide_layouts[6]


def _solid(shape, color, line=None):
    shape.fill.solid(); shape.fill.fore_color.rgb = color
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line; shape.line.width = Pt(1)
    shape.shadow.inherit = False


def rect(slide, x, y, w, h, color, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, I(x), I(y), I(w), I(h))
    _solid(s, color, line); return s


def bg(slide, color):
    rect(slide, -0.1, -0.1, 13.6, 7.7, color)


def accent_tab(slide, x, y):
    rect(slide, x, y, 0.45, 0.07, TEAL)
    rect(slide, x + 0.45, y, 0.45, 0.07, MAROON)


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = 0; tf.margin_right = 0; tf.margin_top = 0; tf.margin_bottom = 0
    return tf


def setpara(p, text, size, color, bold=False, font=BFONT, align=PP_ALIGN.LEFT,
            italic=False, space_after=4):
    p.text = text; p.alignment = align; p.space_after = Pt(space_after)
    r = p.runs[0]; r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.name = font; r.font.color.rgb = color
    return p


def corner_triangle(slide):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, I(0), I(6.55), I(1.05), I(0.95))
    _solid(s, MAROON)
    s.rotation = 0
    # RIGHT_TRIANGLE right-angle at bottom-left by default; flip horizontally
    s.element.spPr.xfrm.set("flipH", "1")


def footer(slide, page):
    tf = textbox(slide, 1.7, 7.06, 9.0, 0.35)
    setpara(tf.paragraphs[0], FOOTER, 9, GREY)
    tf2 = textbox(slide, 12.4, 7.06, 0.7, 0.35)
    setpara(tf2.paragraphs[0], str(page), 9, GREY, align=PP_ALIGN.RIGHT)


def content_slide(title, page, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    bg(s, WHITE)
    rect(s, -0.1, -0.1, 13.6, 0.9, BG)        # top band
    accent_tab(s, 1.0, 1.0)
    tf = textbox(s, 1.0, 1.12, 11.3, 0.9)
    setpara(tf.paragraphs[0], title, 30, DARK, bold=True, font=HFONT)
    y = 2.05
    if subtitle:
        ts = textbox(s, 1.0, 2.02, 11.3, 0.5)
        setpara(ts.paragraphs[0], subtitle, 15, TEAL, bold=True)
        y = 2.55
    corner_triangle(s); footer(s, page)
    return s, y


def bullets(slide, x, y, w, h, items, size=14, gap=6):
    tf = textbox(slide, x, y, w, h)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        lvl = 0; txt = it; bold = False; color = BODY
        if isinstance(it, tuple):
            txt, lvl = it[0], it[1]
            if len(it) > 2: bold = it[2]
            if len(it) > 3: color = it[3]
        setpara(p, ("•  " if lvl == 0 else "–  ") + txt, size,
                color if not bold else DARK, bold=bold, space_after=gap)
        p.level = lvl
        if lvl == 1:
            p.runs[0].font.size = Pt(size - 1)
    return tf


def add_image(slide, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = I(w)
    if h: kw["height"] = I(h)
    return slide.shapes.add_picture(str(path), I(x), I(y), **kw)


def table(slide, x, y, w, h, data, col_w=None, header=True, fs=11):
    rows, cols = len(data), len(data[0])
    gt = slide.shapes.add_table(rows, cols, I(x), I(y), I(w), I(h)).table
    if col_w:
        for j, cw in enumerate(col_w): gt.columns[j].width = I(cw)
    gt.first_row = header
    for i, row in enumerate(data):
        gt.rows[i].height = I(0.34)
        for j, val in enumerate(row):
            c = gt.cell(i, j)
            c.margin_left = I(0.06); c.margin_right = I(0.06)
            c.margin_top = I(0.02); c.margin_bottom = I(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]; p.text = str(val)
            r = p.runs[0]; r.font.size = Pt(fs); r.font.name = BFONT
            r.font.bold = (i == 0 and header)
            if i == 0 and header:
                c.fill.solid(); c.fill.fore_color.rgb = MAROON
                r.font.color.rgb = WHITE
            else:
                c.fill.solid(); c.fill.fore_color.rgb = WHITE if i % 2 else BG
                r.font.color.rgb = DARK
                if j == 0: r.font.bold = True
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
    return gt


# ============================================================ 1. TITLE
s = prs.slides.add_slide(BLANK)
bg(s, BG)
rect(s, -0.1, -0.1, 13.6, 0.55, WHITE)
accent_tab(s, 1.0, 1.7)
tf = textbox(s, 1.0, 1.95, 10.5, 1.6)
setpara(tf.paragraphs[0], "Chess Piece Recognition:", 40, DARK, bold=True, font=HFONT, space_after=0)
setpara(tf.add_paragraph(), "Custom CNN vs YOLO", 40, DARK, bold=True, font=HFONT)
tf2 = textbox(s, 1.0, 4.35, 10, 1.5)
setpara(tf2.paragraphs[0], AUTHOR, 17, BODY, space_after=3)
setpara(tf2.add_paragraph(), COURSE, 17, BODY, space_after=3)
setpara(tf2.add_paragraph(), "Sapienza University of Rome", 17, BODY)

# ============================================================ 2. TOC
s = prs.slides.add_slide(BLANK)
bg(s, BG)
rect(s, -0.1, -0.1, 13.6, 0.55, WHITE)
accent_tab(s, 1.0, 1.0)
tf = textbox(s, 1.0, 1.12, 11, 0.9)
setpara(tf.paragraphs[0], "Table of contents", 30, DARK, bold=True, font=HFONT)
toc = ["Problem & experimental design", "Data pipeline", "Models",
       "Training & hyperparameter tuning", "Results & analysis", "Conclusions"]
for i, t in enumerate(toc):
    col = i % 2; row = i // 2
    x = 1.2 + col * 5.9; y = 2.5 + row * 1.3
    nb = textbox(s, x, y, 0.9, 1.0, MSO_ANCHOR.MIDDLE)
    setpara(nb.paragraphs[0], str(i + 1), 40, DARK, font=HFONT)
    rect(s, x + 0.85, y + 0.12, 0.035, 0.78, MAROON if col == 0 else TEAL)
    lb = textbox(s, x + 1.05, y, 4.6, 1.0, MSO_ANCHOR.MIDDLE)
    setpara(lb.paragraphs[0], t, 15, BODY)
footer(s, 2)

# ============================================================ 3. PROBLEM
s, y = content_slide("Problem & experimental design", 3)
bullets(s, 1.0, y, 11.3, 2.2, [
    ("Goal: recognize chess pieces from a board image; compare a small custom CNN classifier against YOLO detectors on accuracy, latency and compute.", 0),
    ("One shared synthetic dataset, processed into two views so both families see identical data:", 0),
], size=14)
table(s, 1.0, y + 1.35, 11.3, 1.0, [
    ["View", "Consumer", "Unit", "Labels"],
    ["50×50 cell crops", "Custom CNN", "one board cell", "13 (empty + 12 pieces)"],
    ["cell bounding boxes", "YOLO (pico/n/s)", "full board image", "12 piece classes"],
], col_w=[2.6, 2.6, 3.0, 3.1], fs=12)
bullets(s, 1.0, y + 2.6, 11.3, 1.0, [
    ("Fair metric — “occupied-cell accuracy”: each YOLO detection is mapped to its grid cell; the top-confidence class per occupied cell is scored, putting detection on the CNN’s scale.", 0, False, DARK),
], size=13.5)

# ============================================================ 4. DATA
s, y = content_slide("Data pipeline", 4)
add_image(s, FIG / "board_grid.png", 8.5, y + 0.05, w=4.1)
bullets(s, 1.0, y, 7.3, 4.5, [
    ("Source: koryakinp/chess-positions (Kaggle, CC0) — 100k rendered 400×400 boards; label = FEN in the filename; board fills the frame as a clean 8×8 grid (50×50 cells).", 0),
    ("FEN → 8×8 grid → 13-class index per cell (empty=0, white 1-6, black 7-12); occupied cells → YOLO cell-boxes (12-class).", 0),
    ("Working set: 20k train / 4k test boards.", 0),
    ("Class skew: 83% empty, kings 1/board, queens ~0.8%; balanced to ~429k crops (max_empty_ratio).", 0),
    ("Memmap crop cache: precompute (N,64,50,50,3) uint8 → training reads crops with zero JPEG re-decode.", 0),
    ("Kornia differentiable, GPU-batched augmentation (affine/jitter/noise/blur), strength is a tuned hyperparameter.", 0),
], size=12.5, gap=7)

# ============================================================ 5. MODELS
s, y = content_slide("Models", 5)
add_image(s, FIG / "cnn_arch.png", 1.55, 1.98, w=10.2)
ty = 4.65
tf = textbox(s, 1.0, ty - 0.32, 11, 0.3)
setpara(tf.paragraphs[0], "YOLOv8 variants (trained on the same derived boxes):", 13.5, DARK, bold=True)
table(s, 1.0, ty, 11.3, 1.2, [
    ["Variant", "Origin", "Params", "Note"],
    ["yolo_pico", "scaled YAML, from scratch", "184,300", "sub-nano stress test"],
    ["yolov8n", "pretrained (COCO)", "2,692,548", "standard nano"],
    ["yolov8s", "pretrained (COCO)", "9,843,604", "“big” model"],
], col_w=[2.0, 4.3, 2.3, 2.7], fs=12)

# ============================================================ 6. HPO
s, y = content_slide("Hyperparameter tuning (CNN)", 6)
add_image(s, FIG / "hpo.png", 6.7, y + 0.1, w=6.0)
bullets(s, 1.0, y, 5.5, 4.5, [
    ("Optuna study, SQLite storage (resumable), MedianPruner; each trial → a wandb run.", 0),
    ("Search space:", 0, True, DARK),
    ("lr, weight_decay (log); optimizer {adam/adamw/sgd}", 1),
    ("aug_strength 0–0.8; n_blocks 2–4; base_ch {16/32/64}", 1),
    ("fc_dim {64/128/256}; dropout 0–0.5; batch {128/256/512}", 1),
    ("Dual stop: target accuracy OR time budget; ETA predicted from a calibration mini-run.", 0),
], size=12.5, gap=6)
bullets(s, 1.0, 5.5, 11.3, 1.2, [
    ("Insights: AdamW ≫ SGD; heavy augmentation hurts clean renders; the smallest net (28.8k params) won.", 0, False, MAROON),
], size=13)

# ============================================================ 7. TRAINING
s, y = content_slide("Training methodology", 7)
bullets(s, 1.0, y, 11.3, 4.6, [
    ("Engineering: spec → plan → 19-task TDD build; per-task spec + code-quality review; 38 unit/smoke tests.", 0),
    ("Hardware-aware: RTX A4000 16 GB; 90% GPU-memory cap; monitored soft RAM guard (RLIMIT_AS breaks CUDA).", 0),
    ("CNN speed (measured): 0.0283 s/step → ~38 s/epoch → ~142 s per 6-epoch trial; target hit in 1 epoch.", 0),
    ("Two-phase tuning: target 0.995 was met instantly (trial 0) → raised to 0.999 to force real exploration.", 0),
    ("YOLO: imgsz 416, 12 epochs, 25% data fraction; n/s from COCO weights, pico from scratch; optimizer=auto (AdamW).", 0),
    ("Final CNN rebuilt from the best Optuna trial (FixedTrial), retrained, saved, pushed to Hugging Face.", 0),
], size=13.5, gap=9)

# ============================================================ 8. RESULTS TABLE
s, y = content_slide("Results — head-to-head", 8)
table(s, 1.0, y + 0.1, 11.3, 2.6, [
    ["Model", "Occ-acc", "mAP@50-95", "Params", "GFLOPs/bd", "Lat ms/bd", "Disk MB", "Train s"],
    ["custom_cnn ★", "0.9996", "—", "28,813", "0.442", "0.514", "0.125", "~38"],
    ["yolo_pico", "0.2073", "0.255", "184,300", "0.236", "5.491", "0.567", "260"],
    ["yolov8n", "1.0000", "0.995", "2,692,548", "1.469", "4.346", "5.594", "537"],
    ["yolov8s", "1.0000", "0.995", "9,843,604", "4.985", "6.067", "19.918", "734"],
], col_w=[2.1, 1.3, 1.4, 1.8, 1.4, 1.3, 1.2, 1.1], fs=11.5)
bullets(s, 1.0, y + 3.05, 11.3, 1.5, [
    ("Pretrained YOLOs are perfect (grid-aligned boxes are trivial to localize); CNN trails by only 0.04 pp.", 0),
    ("CNN vs yolov8n: ~93× fewer params, ~8.5× lower latency, ~45× smaller — at ~equal accuracy.", 0, False, MAROON),
    ("Pico (sub-nano, from scratch) collapses to 20.7% and is even slower than nano.", 0),
], size=13)

# ============================================================ 9. RESULTS FIGS
s, y = content_slide("Results — accuracy vs cost", 9)
add_image(s, FIG / "acc_vs_params.png", 0.9, y + 0.05, w=6.2)
add_image(s, FIG / "deploy.png", 7.0, y + 0.25, w=6.1)
bullets(s, 1.0, 5.62, 11.3, 1.0, [
    ("Shrink the paradigm, not just the width: a tiny CNN classifier learns in 1 epoch; a tiny detector cannot.", 0, False, DARK),
], size=13.5)

# ============================================================ 10. CONCLUSIONS
s, y = content_slide("Conclusions & recommendations", 10)
bullets(s, 1.0, y, 11.3, 4.4, [
    ("On grid-croppable / embedded targets: ship the custom CNN — 99.96% at 125 KB and 0.5 ms/board.", 0, False, DARK),
    ("For robust real-photo recognition: yolov8n pretrained — perfect here, no board-cropping step, best generalization; yolov8s adds cost without accuracy.", 0),
    ("Do not train a sub-nano YOLO from scratch on small data.", 0),
    ("Trade-off: the CNN’s efficiency assumes an 8×8 grid (free from synthetic geometry); real boards need detection/perspective first — YOLO’s strength.", 0),
    ("Future work: real-photo test set (ChessReD2K), full-data YOLO + HPO sweep, board→FEN reconstruction.", 0),
], size=14, gap=11)

# ============================================================ 11. ARTIFACTS
s, y = content_slide("Artifacts & reproducibility", 11)
bullets(s, 1.0, y, 11.3, 4.2, [
    ("Code: github.com/Honi05/cvproject (TDD, 38 tests, PR #1).", 0),
    ("CNN model: huggingface.co/honi05/chess-piece-cnn", 0),
    ("YOLO models (pico/n/s): huggingface.co/honi05/chess-piece-yolo", 0),
    ("Derived dataset: huggingface.co/datasets/honi05/chess-positions-cv", 0),
    ("Experiment tracking: Weights & Biases — project chess-cnn-vs-yolo (Optuna trials + comparison).", 0),
    ("One command per stage: 01_build_data → 02_train_cnn → train_all_yolo → full_compare → push_artifacts.", 0),
], size=14, gap=10)

# ============================================================ 12. REFERENCES
s, y = content_slide("References", 12)
bullets(s, 1.0, y, 11.3, 3.5, [
    ("Dataset: P. Koryakin, “Chess Positions”, Kaggle (CC0).", 0, True, DARK),
    ("Jocher et al., Ultralytics YOLOv8.", 1),
    ("Akiba et al., “Optuna: A Next-generation Hyperparameter Optimization Framework”, KDD 2019.", 1),
    ("Riba et al., “Kornia: an Open Source Differentiable CV Library for PyTorch”, WACV 2020.", 1),
    ("Slides template: github.com/pietro-nardelli/sapienza-ppt-template (CC BY-NC-SA 4.0).", 1),
], size=13.5, gap=9)

# ============================================================ 13. THANKS
s = prs.slides.add_slide(BLANK)
bg(s, BG)
rect(s, -0.1, -0.1, 13.6, 0.55, WHITE)
accent_tab(s, 1.0, 2.6)
tf = textbox(s, 1.0, 2.9, 11, 1.3)
setpara(tf.paragraphs[0], "Thank you for the attention!", 38, DARK, bold=True, font=HFONT)

prs.save(str(OUT))
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
