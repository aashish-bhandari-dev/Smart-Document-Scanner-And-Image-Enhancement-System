"""Smart Document Scanner & Image Enhancement System.

Streamlit web application with persistent session state and compact typography.
"""

import io
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

from src.preprocessing.image_loader import load_image, SUPPORTED_FORMATS
from src.preprocessing.filters import to_grayscale, apply_gaussian_blur
from src.detection.document_detection import find_document_corners, draw_document_boundary
from src.transformation.perspective import (
    order_points,
    compute_destination_dimensions,
    warp_perspective,
)
from src.enhancement.grayscale import convert_to_grayscale
from src.enhancement.thresholding import adaptive_gaussian_threshold, otsu_threshold
from src.enhancement.denoising import apply_median_filter
from src.enhancement.sharpening import sharpen_image
from src.enhancement.contrast import apply_clahe, adjust_brightness_contrast
from src.export.image_export import export_image_as_bytes
from src.export.pdf_export import export_image_to_pdf
from src.utils.validators import ImageValidationError

# Page Configuration
st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_PATH = Path("assets/sample_images/sample_document.png")

# Scaled typography and compact container styling
COMPACT_UI_CSS = """
<style>
/* Base typography scale */
html, body, [class*="css"] {
    font-size: 0.88rem !important;
}

/* Reduced container padding */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 98% !important;
}

header[data-testid="stHeader"] {
    height: 2rem !important;
}

/* Headings typography */
h1 {
    font-size: 1.55rem !important;
    margin-bottom: 0.2rem !important;
    font-weight: 700 !important;
}
h2 {
    font-size: 1.25rem !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.3rem !important;
    font-weight: 600 !important;
}
h3 {
    font-size: 1.02rem !important;
    margin-top: 0.4rem !important;
    margin-bottom: 0.2rem !important;
    font-weight: 600 !important;
}

/* Metrics scale */
[data-testid="stMetricValue"] {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    padding: 6px 14px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* Image containment to fit screen neatly */
[data-testid="stImage"] img {
    max-height: 48vh !important;
    object-fit: contain !important;
    border-radius: 6px !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
}

/* Description callout boxes */
.info-card {
    background: rgba(99, 102, 241, 0.08);
    border-left: 3px solid #6366f1;
    padding: 8px 14px;
    border-radius: 4px;
    font-size: 0.83rem;
    margin-bottom: 10px;
    color: inherit;
}

/* Action buttons alignment to match uploader */
div[data-testid="stButton"] button {
    height: 42px !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
</style>
"""

st.markdown(COMPACT_UI_CSS, unsafe_allow_html=True)


def format_bytes(size: int) -> str:
    """Format byte count into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def main() -> None:
    # ---------------- SESSION STATE INITIALIZATION ----------------
    # Ensures the uploaded or selected document persists across all slider adjustments and downloads
    if "doc_bytes" not in st.session_state:
        st.session_state["doc_bytes"] = None
    if "doc_filename" not in st.session_state:
        st.session_state["doc_filename"] = None
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # Sidebar
    st.sidebar.title("📄 Document Scanner")
    st.sidebar.caption(
        "Digital Image Processing system that detects physical document boundaries, "
        "eliminates perspective tilt, and enhances legibility using OpenCV."
    )
    st.sidebar.markdown("---")

    # Header
    st.title("Smart Document Scanner & Image Enhancement System")
    st.caption(
        "Upload a photograph of any physical paper document. The computer vision pipeline "
        "automatically isolates boundaries, rectifies perspective distortion, equalizes lighting, "
        "and exports clean archival scans."
    )

    # 1. Image Ingestion Section
    st.subheader("1. Input Document Image")
    
    col_up, col_sample, col_clear = st.columns([3.5, 1.3, 1.0], vertical_alignment="center")

    supported_extensions = [ext.lstrip(".") for ext in sorted(SUPPORTED_FORMATS)]
    with col_up:
        uploader_widget_key = f"file_uploader_{st.session_state['uploader_key']}"
        uploaded_file = st.file_uploader(
            "Upload image file (JPG, JPEG, PNG)",
            type=supported_extensions,
            help="Supported formats: JPG, JPEG, PNG",
            label_visibility="collapsed",
            key=uploader_widget_key
        )

    with col_sample:
        use_sample = st.button(
            "📄 Sample Doc",
            help="Load the bundled sample document photograph",
            use_container_width=True,
            key="btn_sample_doc"
        )

    has_active_doc = st.session_state.get("doc_bytes") is not None
    with col_clear:
        clear_doc = st.button(
            "🗑️ Clear",
            help="Clear currently loaded document",
            use_container_width=True,
            disabled=not has_active_doc,
            key="btn_clear_doc"
        )

    # Handle Upload / Sample / Clear actions
    if clear_doc:
        st.session_state["doc_bytes"] = None
        st.session_state["doc_filename"] = None
        st.session_state["uploader_key"] += 1
        st.rerun()

    if use_sample and SAMPLE_PATH.exists():
        with open(SAMPLE_PATH, "rb") as f:
            st.session_state["doc_bytes"] = f.read()
        st.session_state["doc_filename"] = "sample_document.png"
        st.rerun()

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if file_bytes != st.session_state["doc_bytes"]:
            st.session_state["doc_bytes"] = file_bytes
            st.session_state["doc_filename"] = uploaded_file.name

    # Check if a document is loaded in session state
    if st.session_state["doc_bytes"] is None:
        st.info("👆 Upload an image (JPG, JPEG, PNG) or click **'Sample Doc'** to test the scanner.")
        return

    # Load and decode image from persisted session state bytes
    try:
        buffer = io.BytesIO(st.session_state["doc_bytes"])
        setattr(buffer, "name", st.session_state["doc_filename"])
        bgr_image, metadata = load_image(buffer)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    except ImageValidationError as err:
        st.error(f"Image Validation Error: {err}")
        return
    except Exception as exc:
        st.error(f"Failed to process image: {exc}")
        return

    # Metadata Display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dimensions", f"{metadata['width']} × {metadata['height']} px")
    with col2:
        st.metric("Aspect Ratio", f"{metadata['aspect_ratio']}")
    with col3:
        st.metric("File Format", metadata["format"])
    with col4:
        st.metric("File Size", format_bytes(metadata["file_size_bytes"]))

    st.markdown("---")

    # Sidebar Controls with Clear Explanations
    st.sidebar.subheader("Stage 1: Boundary Detection")
    edge_mode = st.sidebar.radio(
        "Edge Detection Mode",
        options=["Auto Canny (Adaptive)", "Custom Thresholds"],
        index=0,
        help="Auto-Canny dynamically computes upper and lower hysteresis bounds from the median pixel intensity.",
        key="edge_mode_radio"
    )

    low_canny = None
    high_canny = None
    if edge_mode == "Custom Thresholds":
        low_canny = st.sidebar.slider(
            "Canny Low Threshold", 10, 200, 50, step=5,
            help="Lower bound for hysteresis. Weak edges below this are discarded.",
            key="low_canny_slider"
        )
        high_canny = st.sidebar.slider(
            "Canny High Threshold", 50, 300, 150, step=5,
            help="Upper bound for hysteresis. Strong edges above this are retained.",
            key="high_canny_slider"
        )

    working_height = st.sidebar.slider(
        "Working Detection Height (px)",
        min_value=400,
        max_value=1200,
        value=800,
        step=50,
        help="Downsamples image for fast boundary detection. The final transformation re-maps coordinates to full resolution.",
        key="working_height_slider"
    )

    st.sidebar.subheader("Stage 2: Perspective Settings")
    interp_choice = st.sidebar.selectbox(
        "Interpolation Method",
        options=["Bicubic (Sharpest)", "Bilinear (Standard)", "Nearest Neighbor (Fastest)"],
        index=0,
        help="Bicubic uses a 4x4 pixel neighborhood for sharpest text reconstruction without jagged edges.",
        key="interp_choice_select"
    )
    interp_map = {
        "Bicubic (Sharpest)": cv2.INTER_CUBIC,
        "Bilinear (Standard)": cv2.INTER_LINEAR,
        "Nearest Neighbor (Fastest)": cv2.INTER_NEAREST,
    }
    interp_flag = interp_map[interp_choice]

    st.sidebar.subheader("Stage 3: Image Enhancement")
    preset_mode = st.sidebar.selectbox(
        "Enhancement Preset",
        options=[
            "Adaptive B&W Document (Clean Text)",
            "Otsu B&W Document (High Contrast)",
            "Enhanced Grayscale (Equalized)",
            "Sharpened Color Document",
            "Original Rectified Color",
            "Custom Fine-Tuning",
        ],
        index=0,
        help="Choose a pre-tuned enhancement workflow or select Custom Fine-Tuning for full manual slider control.",
        key="preset_mode_select"
    )

    # 1. Boundary Detection Execution
    corners, detection_success, debug_info = find_document_corners(
        bgr_image,
        working_height=working_height,
        low_canny=low_canny,
        high_canny=high_canny,
    )

    # 2. Canonical Corner Ordering
    ordered_corners = order_points(corners)

    # 3. Perspective Warping Execution
    warped_bgr = warp_perspective(bgr_image, ordered_corners, interpolation=interp_flag)
    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    warped_h, warped_w = warped_rgb.shape[:2]

    # Draw Boundary on Original Image
    annotated_rgb = draw_document_boundary(rgb_image, ordered_corners)

    # 4. Enhancement Execution
    enhanced_bgr: np.ndarray
    if preset_mode == "Adaptive B&W Document (Clean Text)":
        gray = convert_to_grayscale(warped_bgr)
        equalized = apply_clahe(gray, clip_limit=2.0)
        sharpened = sharpen_image(equalized, strength=0.8)
        binary = adaptive_gaussian_threshold(sharpened, block_size=15, c_constant=8.0)
        enhanced_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    elif preset_mode == "Otsu B&W Document (High Contrast)":
        gray = convert_to_grayscale(warped_bgr)
        equalized = apply_clahe(gray, clip_limit=2.0)
        binary, _ = otsu_threshold(equalized, pre_blur_ksize=3)
        enhanced_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    elif preset_mode == "Enhanced Grayscale (Equalized)":
        gray = convert_to_grayscale(warped_bgr)
        equalized = apply_clahe(gray, clip_limit=2.5)
        sharpened = sharpen_image(equalized, strength=1.0)
        enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    elif preset_mode == "Sharpened Color Document":
        equalized = apply_clahe(warped_bgr, clip_limit=1.8)
        enhanced_bgr = sharpen_image(equalized, strength=1.0)

    elif preset_mode == "Original Rectified Color":
        enhanced_bgr = warped_bgr.copy()

    else:
        # Custom Manual Adjustments
        st.sidebar.markdown("**Manual Parameter Controls**")
        proc_mode = st.sidebar.radio(
            "Color Mode",
            ["Color", "Grayscale", "Black & White (Binary)"],
            help="Choose output colorspace.",
            key="custom_proc_mode"
        )
        
        brightness = st.sidebar.slider(
            "Brightness", -100, 100, 0, step=5,
            help="Additive offset applied to all pixels.",
            key="custom_brightness"
        )
        contrast = st.sidebar.slider(
            "Contrast", 0.5, 2.5, 1.0, step=0.1,
            help="Multiplicative contrast multiplier (alpha).",
            key="custom_contrast"
        )
        
        apply_cl = st.sidebar.checkbox(
            "Enable CLAHE Equalization", value=True,
            help="Contrast Limited Adaptive Histogram Equalization suppresses shadows.",
            key="custom_apply_cl"
        )
        clahe_limit = 2.0
        if apply_cl:
            clahe_limit = st.sidebar.slider(
                "CLAHE Clip Limit", 1.0, 5.0, 2.0, step=0.5,
                help="Limits contrast amplification to prevent noise buildup.",
                key="custom_clahe_limit"
            )

        apply_sharp = st.sidebar.checkbox(
            "Enable Sharpening", value=True,
            help="Applies a 2D high-pass spatial convolution kernel to sharpen text strokes.",
            key="custom_apply_sharp"
        )
        sharp_strength = 1.0
        if apply_sharp:
            sharp_strength = st.sidebar.slider(
                "Sharpening Strength", 0.0, 3.0, 1.0, step=0.2,
                key="custom_sharp_strength"
            )

        apply_median = st.sidebar.checkbox(
            "Median Noise Filter", value=False,
            help="Non-linear filter that eliminates salt-and-pepper noise without blurring text.",
            key="custom_apply_median"
        )
        median_k = 3
        if apply_median:
            median_k = st.sidebar.slider(
                "Median Kernel Size", 3, 9, 3, step=2,
                key="custom_median_k"
            )

        # Apply adjustments sequentially
        temp_img = adjust_brightness_contrast(warped_bgr, brightness=brightness, contrast=contrast)
        if apply_cl:
            temp_img = apply_clahe(temp_img, clip_limit=clahe_limit)
        if apply_sharp:
            temp_img = sharpen_image(temp_img, strength=sharp_strength)
        if apply_median:
            temp_img = apply_median_filter(temp_img, kernel_size=median_k)

        if proc_mode == "Grayscale":
            gray = convert_to_grayscale(temp_img)
            enhanced_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif proc_mode == "Black & White (Binary)":
            thresh_method = st.sidebar.selectbox(
                "Thresholding Algorithm",
                ["Adaptive Gaussian", "Otsu"],
                help="Adaptive Gaussian calculates local per-pixel thresholds; Otsu calculates a single global bimodal threshold.",
                key="custom_thresh_method"
            )
            if thresh_method == "Adaptive Gaussian":
                b_size = st.sidebar.slider(
                    "Block Size", 3, 31, 15, step=2,
                    help="Neighborhood diameter in pixels used to compute local Gaussian mean.",
                    key="custom_bsize"
                )
                c_val = st.sidebar.slider(
                    "C Value", 1.0, 20.0, 8.0, step=1.0,
                    help="Constant subtracted from mean to fine-tune background separation.",
                    key="custom_cval"
                )
                binary = adaptive_gaussian_threshold(temp_img, block_size=b_size, c_constant=c_val)
            else:
                binary, _ = otsu_threshold(temp_img, pre_blur_ksize=3)
            enhanced_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_bgr = temp_img

    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    # Six Clear, Dedicated Tabs with Explanations
    tabs = st.tabs([
        "1. Original Image",
        "2. Boundary Detection",
        "3. Perspective Correction",
        "4. Image Enhancement",
        "5. Before & After Comparison",
        "6. Export & Download",
    ])

    # ---------------- TAB 1 ----------------
    with tabs[0]:
        st.subheader("Original Uploaded Photograph")
        st.markdown(
            """
            <div class="info-card">
            <b>Original Input Inspection:</b> Displays the raw photographic capture. 
            Mobile camera EXIF rotation tags are verified and normalized so vertical documents are never inverted or rotated.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(rgb_image, caption=f"{metadata['filename']} ({metadata['width']} × {metadata['height']} px)", use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tabs[1]:
        st.subheader("Document Boundary Detection")
        st.markdown(
            """
            <div class="info-card">
            <b>Detection Pipeline:</b> The image is downscaled to a working height while tracking the scale ratio. 
            We apply Gaussian noise smoothing, <b>Canny edge detection</b> with hysteresis, and <b>morphological closing</b> 
            to seal hairline gaps. Contours are extracted and simplified via <b>Douglas-Peucker polygon approximation</b> 
            (<code>cv2.approxPolyDP</code>) into a strictly convex 4-corner quadrilateral.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if detection_success:
            st.success(f"✅ {debug_info['message']} (Downsample ratio: {debug_info['ratio']:.2f})")
        else:
            st.warning(f"⚠️ {debug_info['message']}")

        col_det1, col_det2 = st.columns(2)
        with col_det1:
            st.markdown("**Detected Document Quadrilateral**")
            st.image(annotated_rgb, caption="Green polygon = detected document; Red markers = corners", use_container_width=True)

        with col_det2:
            st.markdown("**Binary Edge Maps**")
            subcol_e1, subcol_e2 = st.columns(2)
            with subcol_e1:
                st.image(debug_info["edge_map"], caption="Canny Edge Map", use_container_width=True)
            with subcol_e2:
                st.image(debug_info["closed_edges"], caption="Morphological Closed Map", use_container_width=True)

            st.markdown("**Canonical Corner Coordinates (Full-Resolution Space)**")
            labels = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Right (BR)", "Bottom-Left (BL)"]
            for label, pt in zip(labels, ordered_corners):
                st.write(f"- **{label}**: `(x={pt[0]:.1f}, y={pt[1]:.1f})`")

    # ---------------- TAB 3 ----------------
    with tabs[2]:
        st.subheader("Perspective Transformation (Orthographic Rectification)")
        st.markdown(
            """
            <div class="info-card">
            <b>Homography Warping:</b> Perspective distortion is mathematically corrected. 
            Opposite edge Euclidean distances determine the physical width and height to prevent stretching. 
            A 3×3 projective homography matrix <code>M</code> maps the angled quadrilateral into an orthogonal plane 
            using high-order <b>Bicubic interpolation</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_dim1, col_dim2, col_dim3 = st.columns(3)
        with col_dim1:
            st.metric("Rectified Width", f"{warped_w} px")
        with col_dim2:
            st.metric("Rectified Height", f"{warped_h} px")
        with col_dim3:
            st.metric("Output Aspect Ratio", f"{round(warped_w / float(warped_h), 3)}")

        col_warp1, col_warp2 = st.columns(2)
        with col_warp1:
            st.markdown("**Angled Photograph with Boundary Overlay**")
            st.image(annotated_rgb, use_container_width=True)
        with col_warp2:
            st.markdown("**Rectified Top-Down Document**")
            st.image(warped_rgb, caption=f"Rectified ({warped_w}×{warped_h} px)", use_container_width=True)

    # ---------------- TAB 4 ----------------
    with tabs[3]:
        st.subheader("Image Enhancement")
        st.markdown(
            f"""
            <div class="info-card">
            <b>Enhancement Engine:</b> Active Preset is <b>{preset_mode}</b>. 
            <b>CLAHE</b> equalizes lighting gradients across the paper; 
            <b>Adaptive Gaussian Thresholding</b> binarizes text by computing local neighborhood averages to overcome shadows; 
            <b>High-pass sharpening</b> enhances character edge contrast.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(enhanced_rgb, caption=f"Enhanced Output ({preset_mode})", use_container_width=True)

    # ---------------- TAB 5 ----------------
    with tabs[4]:
        st.subheader("Before-and-After Comparison")
        st.markdown(
            """
            <div class="info-card">
            <b>Progression Overview:</b> Step-by-step visual comparison from raw capture to rectified flat plane, 
            and finally to the enhanced, shadow-free scanned document.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_comp1, col_comp2, col_comp3 = st.columns(3)
        with col_comp1:
            st.markdown("**1. Original Photograph**")
            st.image(rgb_image, use_container_width=True)
        with col_comp2:
            st.markdown("**2. Perspective Rectified**")
            st.image(warped_rgb, use_container_width=True)
        with col_comp3:
            st.markdown("**3. Final Enhanced Scan**")
            st.image(enhanced_rgb, use_container_width=True)

    # ---------------- TAB 6 ----------------
    with tabs[5]:
        st.subheader("Document Export & Download")
        st.markdown(
            """
            <div class="info-card">
            <b>Export Subsystem:</b> Download your processed document in digital image formats (PNG/JPEG) 
            or compile it into a standard vector-wrapped PDF document with aspect ratio retention.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            st.markdown("### 🖼️ PNG Image")
            st.caption("Lossless compression. Ideal for archiving text with sharp character edges.")
            png_bytes, png_mime = export_image_as_bytes(enhanced_bgr, format="PNG")
            st.download_button(
                label="Download PNG",
                data=png_bytes,
                file_name="scanned_document.png",
                mime=png_mime,
                use_container_width=True,
                key="download_png_btn"
            )

        with col_exp2:
            st.markdown("### 📷 JPEG Image")
            st.caption("Compact file size with adjustable lossy compression.")
            jpeg_quality = st.slider(
                "JPEG Quality", 50, 100, 95, step=5,
                key="export_jpeg_quality_slider"
            )
            jpeg_bytes, jpeg_mime = export_image_as_bytes(enhanced_bgr, format="JPEG", quality=jpeg_quality)
            st.download_button(
                label=f"Download JPEG ({jpeg_quality}%)",
                data=jpeg_bytes,
                file_name="scanned_document.jpg",
                mime=jpeg_mime,
                use_container_width=True,
                key="download_jpeg_btn"
            )

        with col_exp3:
            st.markdown("### 📑 PDF Document")
            st.caption("Archival PDF document preserving physical document proportions.")
            pdf_page_size = st.selectbox(
                "Page Format", ["A4", "Letter", "Fit"], index=0,
                help="A4 and Letter fit the scan centered with margins; Fit matches exact pixel dimensions.",
                key="export_pdf_pagesize_select"
            )
            pdf_bytes = export_image_to_pdf(enhanced_bgr, page_size=pdf_page_size)
            st.download_button(
                label=f"Download PDF ({pdf_page_size})",
                data=pdf_bytes,
                file_name="scanned_document.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_btn"
            )


if __name__ == "__main__":
    main()
