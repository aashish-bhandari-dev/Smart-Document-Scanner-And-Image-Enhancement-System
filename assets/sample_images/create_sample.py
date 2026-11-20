"""Utility script to generate realistic, professional document photographs for testing."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_realistic_invoice(doc_w: int = 700, doc_h: int = 950) -> np.ndarray:
    """Generate a clean, high-resolution official invoice document with text and tables."""
    # Create white paper with subtle warm tone
    doc_pil = Image.new("RGB", (doc_w, doc_h), (252, 252, 250))
    draw = ImageDraw.Draw(doc_pil)

    # Use standard default font
    try:
        font_title = ImageFont.truetype("Arial.ttf", 26)
        font_h2 = ImageFont.truetype("Arial.ttf", 16)
        font_body = ImageFont.truetype("Arial.ttf", 13)
        font_small = ImageFont.truetype("Arial.ttf", 11)
        font_bold = ImageFont.truetype("Arial.ttf", 13)
    except IOError:
        font_title = ImageFont.load_default()
        font_h2 = ImageFont.load_default()
        font_body = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    # Outer decorative header line
    draw.rectangle([(40, 40), (doc_w - 40, 44)], fill=(30, 41, 59))

    # Header section
    draw.text((40, 58), "APEX DIGITAL LABS INC.", fill=(15, 23, 42), font=font_title)
    draw.text((40, 95), "Computer Vision & Advanced Document Processing", fill=(71, 85, 105), font=font_small)
    draw.text((40, 112), "100 Innovation Parkway, Suite 500 | contact@apexdigital.tech", fill=(100, 116, 139), font=font_small)

    # Invoice Tag on top right
    draw.text((doc_w - 200, 58), "TAX INVOICE", fill=(30, 41, 59), font=font_title)
    draw.text((doc_w - 200, 95), "Invoice #: INV-2026-8842", fill=(51, 65, 85), font=font_body)
    draw.text((doc_w - 200, 115), "Date: November 18, 2026", fill=(51, 65, 85), font=font_body)
    draw.text((doc_w - 200, 135), "Status: PAID IN FULL", fill=(16, 149, 93), font=font_body)

    # Horizontal divider
    draw.line([(40, 160), (doc_w - 40, 160)], fill=(203, 213, 225), width=1)

    # Bill To Section
    draw.text((40, 175), "BILLED TO:", fill=(100, 116, 139), font=font_small)
    draw.text((40, 195), "Global Research Institute", fill=(15, 23, 42), font=font_h2)
    draw.text((40, 218), "Department of Computational Intelligence & Robotics", fill=(51, 65, 85), font=font_body)
    draw.text((40, 236), "742 University Crescent, Cambridge, MA 02138", fill=(51, 65, 85), font=font_body)

    # Table Header Box
    table_top = 280
    draw.rectangle([(40, table_top), (doc_w - 40, table_top + 32)], fill=(241, 245, 249))
    draw.line([(40, table_top), (doc_w - 40, table_top)], fill=(148, 163, 184), width=1)
    draw.line([(40, table_top + 32), (doc_w - 40, table_top + 32)], fill=(148, 163, 184), width=1)

    draw.text((55, table_top + 8), "ITEM DESCRIPTION", fill=(30, 41, 59), font=font_bold)
    draw.text((360, table_top + 8), "QTY", fill=(30, 41, 59), font=font_bold)
    draw.text((440, table_top + 8), "RATE", fill=(30, 41, 59), font=font_bold)
    draw.text((doc_w - 120, table_top + 8), "TOTAL", fill=(30, 41, 59), font=font_bold)

    # Table Items
    items = [
        ("01", "Edge Detection & Contour Extraction Module", "1", "$850.00", "$850.00"),
        ("02", "Projective Homography Rectification Core", "1", "$620.00", "$620.00"),
        ("03", "Adaptive Gaussian & Otsu Binarization Engine", "1", "$480.00", "$480.00"),
        ("04", "CIELAB CLAHE Contrast Equalization Service", "1", "$350.00", "$350.00"),
        ("05", "Vector PDF Document Compiler Subsystem", "1", "$420.00", "$420.00"),
        ("06", "High-Resolution Benchmark Testing & CI Validation", "1", "$280.00", "$280.00"),
    ]

    curr_y = table_top + 45
    for idx, (code, desc, qty, rate, total) in enumerate(items):
        # Alternating row background
        if idx % 2 == 1:
            draw.rectangle([(40, curr_y - 6), (doc_w - 40, curr_y + 26)], fill=(248, 250, 252))
        
        draw.text((55, curr_y), f"{code}. {desc}", fill=(30, 41, 59), font=font_body)
        draw.text((370, curr_y), qty, fill=(51, 65, 85), font=font_body)
        draw.text((440, curr_y), rate, fill=(51, 65, 85), font=font_body)
        draw.text((doc_w - 120, curr_y), total, fill=(15, 23, 42), font=font_body)
        
        # Subtle row separator
        draw.line([(40, curr_y + 26), (doc_w - 40, curr_y + 26)], fill=(226, 232, 240), width=1)
        curr_y += 36

    # Totals Section
    totals_y = curr_y + 20
    draw.text((doc_w - 240, totals_y), "Subtotal:", fill=(71, 85, 105), font=font_body)
    draw.text((doc_w - 120, totals_y), "$3,000.00", fill=(15, 23, 42), font=font_body)

    draw.text((doc_w - 240, totals_y + 25), "Tax (8.25%):", fill=(71, 85, 105), font=font_body)
    draw.text((doc_w - 120, totals_y + 25), "$247.50", fill=(15, 23, 42), font=font_body)

    draw.rectangle([(doc_w - 250, totals_y + 55), (doc_w - 40, totals_y + 90)], fill=(241, 245, 249))
    draw.text((doc_w - 240, totals_y + 65), "TOTAL DUE:", fill=(15, 23, 42), font=font_bold)
    draw.text((doc_w - 120, totals_y + 65), "$3,247.50", fill=(15, 23, 42), font=font_title)

    # Terms & Notes
    notes_y = totals_y + 110
    draw.line([(40, notes_y), (doc_w - 40, notes_y)], fill=(203, 213, 225), width=1)
    draw.text((40, notes_y + 15), "TERMS & CONDITIONS", fill=(100, 116, 139), font=font_small)
    draw.text((40, notes_y + 35), "1. All deliverables were verified and approved according to specification DIP-2026.", fill=(71, 85, 105), font=font_small)
    draw.text((40, notes_y + 52), "2. Payments processed automatically via ACH Corporate Transfer.", fill=(71, 85, 105), font=font_small)
    draw.text((40, notes_y + 69), "3. Direct billing questions to: billing@apexdigital.tech | Tel: +1 (800) 555-0199.", fill=(71, 85, 105), font=font_small)

    # Signature and Stamp Box
    draw.line([(doc_w - 220, notes_y + 80), (doc_w - 50, notes_y + 80)], fill=(71, 85, 105), width=1)
    draw.text((doc_w - 210, notes_y + 85), "Authorized Signature", fill=(100, 116, 139), font=font_small)

    return cv2.cvtColor(np.array(doc_pil), cv2.COLOR_RGB2BGR)


def generate_sample_document(output_path: str = "assets/sample_images/sample_document.png") -> None:
    """Compose the document onto a photographic background with realistic camera angle perspective."""
    doc_w, doc_h = 700, 950
    doc_bgr = create_realistic_invoice(doc_w, doc_h)

    # Canvas representing dark wooden desk / office surface (1200 x 1400)
    canvas_w, canvas_h = 1200, 1400
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Dark textured desk gradient
    for y in range(canvas_h):
        shade = int(32 + 18 * (y / canvas_h))
        canvas[y, :] = (shade, shade - 2, shade - 4)

    # Add realistic desk grain noise
    desk_noise = np.random.normal(0, 4, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + desk_noise, 0, 255).astype(np.uint8)

    # Document corners (flat source)
    src_pts = np.float32([
        [0, 0],              # TL
        [doc_w - 1, 0],      # TR
        [doc_w - 1, doc_h - 1],  # BR
        [0, doc_h - 1]       # BL
    ])

    # Realistic camera capture perspective (tilted portrait document on desk)
    dst_pts = np.float32([
        [220, 140],    # Top-Left (skewed perspective)
        [960, 190],    # Top-Right
        [1060, 1260],  # Bottom-Right (wider in foreground)
        [120, 1190],   # Bottom-Left
    ])

    # 1. Soft Drop Shadow
    shadow_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    # Offset shadow slightly down and right
    shadow_pts = dst_pts + np.array([12, 16], dtype=np.float32)
    cv2.fillConvexPoly(shadow_mask, shadow_pts.astype(np.int32), 160)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (55, 55), 0)

    # Darken canvas where shadow falls
    shadow_factor = 1.0 - (shadow_mask.astype(np.float32) / 255.0) * 0.55
    for c in range(3):
        canvas[:, :, c] = np.clip(canvas[:, :, c].astype(np.float32) * shadow_factor, 0, 255).astype(np.uint8)

    # 2. Warp document onto canvas
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc_bgr, matrix, (canvas_w, canvas_h), flags=cv2.INTER_CUBIC)

    # 3. Alpha blend document onto desk
    doc_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillConvexPoly(doc_mask, dst_pts.astype(np.int32), 255)
    doc_mask_inv = cv2.bitwise_not(doc_mask)

    bg_part = cv2.bitwise_and(canvas, canvas, mask=doc_mask_inv)
    final_img = cv2.add(bg_part, warped_doc)

    # Add slight camera lens blur / vignette
    final_img = cv2.GaussianBlur(final_img, (3, 3), 0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, final_img)
    print(f"Realistic sample document generated successfully at: {output_path}")


if __name__ == "__main__":
    generate_sample_document()
