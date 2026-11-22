"""Generate a high-resolution one-page Arabic/English presentation poster JPG for the professor."""

from __future__ import annotations

import shutil
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont


def ar(text: str) -> str:
    """Reshape Arabic strings and apply BiDi algorithm with explicit RTL base direction."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped, base_dir="R")


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    font_dir = Path("C:/Windows/Fonts")
    reg_path = font_dir / "tahoma.ttf" if (font_dir / "tahoma.ttf").exists() else font_dir / "arial.ttf"
    bold_path = font_dir / "tahomabd.ttf" if (font_dir / "tahomabd.ttf").exists() else font_dir / "arialbd.ttf"

    return {
        "title": ImageFont.truetype(str(bold_path), 44),
        "subtitle": ImageFont.truetype(str(bold_path), 26),
        "badge": ImageFont.truetype(str(bold_path), 22),
        "pitch": ImageFont.truetype(str(bold_path), 23),
        "card_title": ImageFont.truetype(str(bold_path), 27),
        "body_bold": ImageFont.truetype(str(bold_path), 21),
        "body": ImageFont.truetype(str(reg_path), 20),
        "small": ImageFont.truetype(str(reg_path), 18),
        "footer": ImageFont.truetype(str(bold_path), 19),
    }


def draw_rtl_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[tuple[str, str, str]],
    right_x: int,
    start_y: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    line_spacing: int = 11,
) -> int:
    """Draw right-aligned Arabic/mixed lines. Each item is (text, font_key, color)."""
    y = start_y
    for text, fkey, color in lines:
        font = fonts[fkey]
        shaped = ar(text)
        bbox = draw.textbbox((0, 0), shaped, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text((right_x - w, y), shaped, font=font, fill=color)
        y += max(h, 24) + line_spacing
    return y


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    header_color: str,
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> int:
    """Draw a rounded card with a colored top header bar and right-aligned Arabic title."""
    x0, y0, x1, y1 = box
    # Drop shadow
    draw.rounded_rectangle((x0 + 4, y0 + 5, x1 + 4, y1 + 5), radius=18, fill="#cbd5e1")
    # Card background
    draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#cbd5e1", width=2)
    # Header bar
    draw.rounded_rectangle((x0, y0, x1, y0 + 58), radius=18, fill=header_color)
    draw.rectangle((x0, y0 + 38, x1, y0 + 58), fill=header_color)

    shaped_title = ar(title)
    bbox = draw.textbbox((0, 0), shaped_title, font=fonts["card_title"])
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x1 - 24 - tw, y0 + (54 - th) // 2 - 2), shaped_title, font=fonts["card_title"], fill="#ffffff")
    return y0 + 74


def main() -> None:
    W, H = 1800, 2450
    canvas = Image.new("RGB", (W, H), "#f1f5f9")
    draw = ImageDraw.Draw(canvas)
    fonts = load_fonts()

    # 1. Top Header Banner
    draw.rectangle((0, 0, W, 195), fill="#0f172a")
    draw.rectangle((0, 188, W, 195), fill="#10b981")

    title_str = ar("ملخص مشروع: EuroSAT-Sentinel (المرحلة الأولى - Phase 1)")
    tb = draw.textbbox((0, 0), title_str, font=fonts["title"])
    draw.text(((W - (tb[2] - tb[0])) // 2, 28), title_str, font=fonts["title"], fill="#ffffff")

    sub_str = ar("الذكاء الاصطناعي التفسيري (Explainable Deep Learning & Computer Vision) لتحليل صور الأقمار الصناعية ومراقبة البيئة")
    sb = draw.textbbox((0, 0), sub_str, font=fonts["subtitle"])
    draw.text(((W - (sb[2] - sb[0])) // 2, 95), sub_str, font=fonts["subtitle"], fill="#38bdf8")

    # Metric Badges in Header
    badges = [
        ("Top-1 Test Acc: 96.22%", "#059669"),
        ("Top-3 Test Acc: 100.0%", "#0284c7"),
        ("Macro F1-Score: 0.9622", "#7c3aed"),
        ("DL Gain vs Classical ML: +19.55%", "#d97706"),
    ]
    bx = 70
    for b_text, b_col in badges:
        draw.rounded_rectangle((bx, 142, bx + 400, 178), radius=10, fill=b_col)
        bb = draw.textbbox((0, 0), b_text, font=fonts["badge"])
        draw.text((bx + (400 - (bb[2] - bb[0])) // 2, 147), b_text, font=fonts["badge"], fill="#ffffff")
        bx += 420

    # 2. Elevator Pitch Banner (Full Width)
    pitch_box = (45, 220, W - 45, 385)
    draw.rounded_rectangle((49, 225, W - 41, 390), radius=16, fill="#cbd5e1")
    draw.rounded_rectangle(pitch_box, radius=16, fill="#ecfdf5", outline="#10b981", width=3)

    pitch_lines = [
        ("1. تقديم الفكرة في 30 ثانية للدكتور (Elevator Pitch):", "card_title", "#065f46"),
        ("«يا دكتور، مشروعنا اسمه (EuroSAT-Sentinel)، وهو نظام ذكاء اصطناعي قائم على الـ Deep Learning والـ Computer Vision", "pitch", "#0f172a"),
        ("لتحليل صور القمر الصناعي الأوروبي (Sentinel-2) وتصنيف استخدامات الأراضي والغطاء البيئي إلى 10 فئات مختلفة بدقة 96.22%،", "pitch", "#0f172a"),
        ("مع تفسير بصري لقرار الموديل باستخدام خرائط (Grad-CAM) وتقديم توصيات بيئية ذكية في تطبيق تفاعلي.»", "pitch", "#065f46"),
    ]
    draw_rtl_lines(draw, pitch_lines, right_x=W - 75, start_y=235, fonts=fonts, line_spacing=10)

    # Define 2-Column Layout
    # Right Column (Arabic reading order starts on the right): x from 920 to 1755
    # Left Column: x from 45 to 880
    R_X0, R_X1 = 920, W - 45
    L_X0, L_X1 = 45, 880

    # Card R1: Problem & Motivation (Top Right)
    y_start = draw_card(draw, (R_X0, 410, R_X1, 790), "2. المشكلة الحقيقية والهدف (Problem & Motivation)", "#1e3a8a", fonts)
    r1_lines = [
        ("• المشكلة الواقعية:", "body_bold", "#1e3a8a"),
        ("  متابعة التغيرات البيئية مثل إزالة الغابات، الزحف العمراني على الأراضي", "body", "#1e293b"),
        ("  الزراعية، ومراقبة المسطحات المائية تعتمد على صور الأقمار الصناعية", "body", "#1e293b"),
        ("  التي تنتج ملايين الصور يومياً، وتحليلها يدوياً بطيء ومكلف جداً.", "body", "#1e293b"),
        ("• الهدف من المشروع:", "body_bold", "#0f766e"),
        ("  بناء نظام آلي، سريع، وقابل للتفسير (Explainable AI) يصنف أي منطقة", "body", "#1e293b"),
        ("  من صور القمر الصناعي Sentinel-2 ويوضح بصرياً سبب اتخاذ القرار", "body", "#1e293b"),
        ("  لمساعدة الباحثين وجهات التخطيط العمراني وحماية البيئة.", "body", "#1e293b"),
    ]
    draw_rtl_lines(draw, r1_lines, right_x=R_X1 - 24, start_y=y_start, fonts=fonts, line_spacing=10)

    # Card L1: Why Deep Learning is Necessary (Top Left)
    y_start = draw_card(draw, (L_X0, 410, L_X1, 955), "3. لماذا نحتاج الـ Deep Learning تحديداً؟ (سؤال جوهري)", "#0f766e", fonts)
    l1_lines = [
        ("• تشابه الألوان في صور الأقمار الصناعية:", "body_bold", "#0f766e"),
        ("  لو اعتمدنا على الألوان فقط (Color Histograms)، الموديل سيفشل في التفريق", "body", "#1e293b"),
        ("  بين الفئات المتشابهة لونياً والمختلفة هندسياً:", "body", "#1e293b"),
        ("  1) النهر (River) والبحيرة (SeaLake): نفس لون المياه، لكن الفرق في", "body", "#1e293b"),
        ("     الشكل الهندسي (مجرى طولي متعرج وضفاف مقابل مسطح مائي مفتوح).", "body", "#1e293b"),
        ("  2) الطرق (Highway) والمناطق السكنية (Residential) والصناعية (Industrial):", "body", "#1e293b"),
        ("     كلها تشترك في اللون الرمادي، والفرق في التوزيع المكاني (Spatial Topology).", "body", "#1e293b"),
        ("• دور الـ CNNs (ResNet-18 Transfer Learning):", "body_bold", "#1e3a8a"),
        ("  تتعلم خصائص مكانية هرمية (Hierarchical Spatial Features) من الحواف", "body", "#1e293b"),
        ("  إلى تعرجات الأنهار وتخطيط المباني والمزارع.", "body", "#1e293b"),
        ("• إثبات عملي بالتجربة في مشروعنا:", "body_bold", "#b45309"),
        ("  - نموذج Classical ML (Random Forest): حقق دقة 76.67% فقط.", "body_bold", "#dc2626"),
        ("  - نموذج Deep Learning (ResNet-18): حقق دقة 96.22% (+19.55% فرق).", "body_bold", "#059669"),
    ]
    draw_rtl_lines(draw, l1_lines, right_x=L_X1 - 24, start_y=y_start, fonts=fonts, line_spacing=9)

    # Card R2: Dataset & Methodology (Middle Right)
    y_start = draw_card(draw, (R_X0, 815, R_X1, 1390), "4. الداتا والمنهجية العلمية (Dataset & Methodology)", "#0369a1", fonts)
    r2_lines = [
        ("• أولاً - الداتا المعتمدة (EuroSAT RGB Benchmark):", "body_bold", "#0369a1"),
        ("  صور حقيقية من القمر الصناعي Sentinel-2 تغطي 34 دولة مقسمة إلى", "body", "#1e293b"),
        ("  10 فئات متوازنة (غابات، محاصيل موسمية/دائمة، مراعي، أنهار، طرق، سكن، صناعة).", "body", "#1e293b"),
        ("• ثانياً - تقسيم صارم بدون تسريب بيانات (Zero Data Leakage):", "body_bold", "#0369a1"),
        ("  تقسيم Stratified متوازن (70% تدريب / 15% تحقق / 15% اختبار) مع اختبارات", "body", "#1e293b"),
        ("  آلية (8/8 Pytest Passing) تضمن عدم تكرار أي صورة بين التقسيمات.", "body", "#1e293b"),
        ("• ثالثاً - Data Augmentation مخصص لصور الأقمار الصناعية:", "body_bold", "#0369a1"),
        ("  الصور الملتقطة من أعلى (Overhead) ليس لها اتجاه علوي ثابت، فطبقنا", "body", "#1e293b"),
        ("  دورانات 90 درجة وانعكاسات أفقية ورأسية (Flips & Rotations) و Color Jitter.", "body", "#1e293b"),
        ("• رابعاً - استراتيجية التدريب على مرحلتين (Two-Stage Transfer Learning):", "body_bold", "#0369a1"),
        ("  - المرحلة 1 (Warmup): تجميد الـ Backbone وتدريب طبقة التصنيف أولاً.", "body", "#1e293b"),
        ("  - المرحلة 2 (Fine-Tuning): تدريب الشبكة بالكامل بـ AdamW و Cosine LR", "body", "#1e293b"),
        ("    مع Label Smoothing (0.05) لمنع الـ Overfitting.", "body", "#1e293b"),
    ]
    draw_rtl_lines(draw, r2_lines, right_x=R_X1 - 24, start_y=y_start, fonts=fonts, line_spacing=9)

    # Card L2: Results & Interactive Demo (Middle Left)
    y_start = draw_card(draw, (L_X0, 980, L_X1, 1390), "5. النتائج الفعلية والتطبيق التفاعلي (Results & Demo)", "#6d28d9", fonts)
    l2_lines = [
        ("• نتائج الاختبار النهائية (Held-Out Test Set):", "body_bold", "#6d28d9"),
        ("  - دقة التصنيف: Top-1 Accuracy: 96.22%  |  Top-3 Accuracy: 100.0%", "body_bold", "#059669"),
        ("  - معيار Macro F1-Score: 0.9622 (ودقة 100% في اكتشاف الغابات Forest).", "body", "#1e293b"),
        ("• التفسير البصري والتطبيق التفاعلي (Gradio + Grad-CAM):", "body_bold", "#6d28d9"),
        ("  1) يرفع الدكتور أي صورة قمر صناعي على واجهة الويب التفاعلية.", "body", "#1e293b"),
        ("  2) تظهر خريطة حرارية (Grad-CAM Heatmap) تضيء باللون الأحمر والأصفر", "body", "#1e293b"),
        ("     على العنصر الذي اعتمد عليه الموديل (مثل مسار النهر أو الطريق السريع).", "body", "#1e293b"),
        ("  3) يصدر النظام تقريراً بيئياً ذكياً (Ecological Advisory) بالتوصيات.", "body", "#1e293b"),
    ]
    draw_rtl_lines(draw, l2_lines, right_x=L_X1 - 24, start_y=y_start, fonts=fonts, line_spacing=10)

    # Card Full-Width: Phase 2 & Phase 3 Roadmap + Presentation Tip
    y_start = draw_card(draw, (45, 1415, W - 45, 1695), "6. خطة التطوير المستقبلية (Phase 2 & Phase 3) ونصيحة العرض", "#b45309", fonts)
    road_lines = [
        ("• في Phase 2 (تحسين الموديل): دمج الـ 13 قناة طيفية (13-Band Multispectral) لحساب مؤشرات النباتات والمياه (NDVI & NDWI)", "body_bold", "#1e293b"),
        ("  لفصل المحاصيل المتشابهة بدقة أعلى، ومقارنة الـ CNNs مع الـ Vision Transformers (ViT / Swin Transformer).", "body", "#1e293b"),
        ("• في Phase 3 (التطبيق المتقدم): مسح خرائط جغرافية كبيرة كاملة (Sliding-Window Semantic Mapping) مع نظام إنذار مبكر", "body_bold", "#1e293b"),
        ("  يقارن صورتين في زمنين مختلفين لاكتشاف إزالة الغابات أو الزحف العمراني تلقائياً.", "body", "#1e293b"),
        ("• نصيحة ذهبية أثناء المناقشة: اعرضي للدكتور الصور بالأسفل (مقارنة الـ Deep Learning ضد الـ Classical ML وخرائط الـ Grad-CAM) وشغلي التطبيق التفاعلي!", "body_bold", "#b45309"),
    ]
    draw_rtl_lines(draw, road_lines, right_x=W - 75, start_y=y_start, fonts=fonts, line_spacing=10)

    # 7. Embedded Visual Proof Strip at the Bottom (Real Project Figures)
    vis_box = (45, 1720, W - 45, 2365)
    draw.rounded_rectangle((49, 1725, W - 41, 2370), radius=18, fill="#cbd5e1")
    draw.rounded_rectangle(vis_box, radius=18, fill="#ffffff", outline="#94a3b8", width=2)

    vis_header = ar("7. نماذج من النتائج والرسومات الفعلية المنفذة في المشروع (Actual Executed Visualizations)")
    vh_box = draw.textbbox((0, 0), vis_header, font=fonts["subtitle"])
    draw.text(((W - (vh_box[2] - vh_box[0])) // 2, 1735), vis_header, font=fonts["subtitle"], fill="#0f172a")

    fig1_path = Path("outputs/figures/per_class_f1_and_baseline.png")
    fig2_path = Path("outputs/figures/gradcam_explainability_gallery.png")

    if fig1_path.exists():
        img1 = Image.open(fig1_path).convert("RGB")
        img1.thumbnail((820, 540), Image.Resampling.LANCZOS)
        canvas.paste(img1, (70, 1795))

    if fig2_path.exists():
        img2 = Image.open(fig2_path).convert("RGB")
        img2.thumbnail((820, 540), Image.Resampling.LANCZOS)
        canvas.paste(img2, (910, 1795))

    # Footer Bar
    draw.rectangle((0, H - 60, W, H), fill="#0f172a")
    footer_text = "GitHub Repository: https://github.com/mayarmajed2/EuroSAT-Sentinel   |   University Deep Learning & Computer Vision Project (Phase 1)"
    fb = draw.textbbox((0, 0), footer_text, font=fonts["footer"])
    draw.text(((W - (fb[2] - fb[0])) // 2, H - 42), footer_text, font=fonts["footer"], fill="#e2e8f0")

    out_repo_jpg = Path("outputs/figures/professor_presentation_summary_ar.jpg")
    out_repo_jpg.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_repo_jpg, "JPEG", quality=95, optimize=True)

    artifact_jpg = Path(
        "C:/Users/Mayoura/.gemini/antigravity-cli/brain/44435c7f-330f-4235-b430-0afb9db2f724/professor_presentation_summary_ar.jpg"
    )
    artifact_jpg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_repo_jpg, artifact_jpg)
    print(f"Saved one-page summary JPG to {out_repo_jpg} and {artifact_jpg}")


if __name__ == "__main__":
    main()
