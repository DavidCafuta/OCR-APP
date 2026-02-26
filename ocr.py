import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
from pdf2image import convert_from_bytes
import pytesseract
import easyocr

st.set_page_config(page_title="OCR HR/EN", page_icon="🔎")
st.title("🔎 OCR (Croatian + English)")
st.markdown("Upload a photo or PDF and extract text (Croatian/English).")


def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(pil_img, upscale=1.5, do_threshold=True):
    cv_img = pil_to_cv(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    if upscale and upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    if do_threshold:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th)
    return Image.fromarray(gray)


def ocr_tesseract(pil_img, lang_codes="hrv+eng", psm=6):
    config = f"-l {lang_codes} --psm {psm}"
    return pytesseract.image_to_string(pil_img, config=config)


@st.cache_resource
def get_easyocr_reader(langs_tuple):
    return easyocr.Reader(list(langs_tuple), gpu=False)


def ocr_easyocr(pil_img, langs=("hr", "en")):
    reader = get_easyocr_reader(tuple(langs))
    lines = reader.readtext(np.array(pil_img), detail=0, paragraph=True)
    return "\n".join(lines)


def run_ocr(pil_img, engine, lang_hr, lang_en, psm):
    """Run chosen engine with automatic fallback to EasyOCR."""
    langs_easy = []
    if lang_hr:
        langs_easy.append("hr")
    if lang_en:
        langs_easy.append("en")

    if not langs_easy:
        st.error("Select at least one language (Croatian and/or English).")
        st.stop()

    if engine == "Tesseract":
        tlangs = []
        if lang_hr:
            tlangs.append("hrv")
        if lang_en:
            tlangs.append("eng")
        try:
            return ocr_tesseract(pil_img, lang_codes="+".join(tlangs), psm=psm)
        except Exception as e:
            st.warning(f"Tesseract failed ({e}), falling back to EasyOCR.")
            return ocr_easyocr(pil_img, langs=tuple(langs_easy))
    else:
        return ocr_easyocr(pil_img, langs=tuple(langs_easy))


# --- UI ---
uploaded_file = st.file_uploader(
    "Upload image or PDF (JPG, PNG, PDF)",
    type=["jpg", "jpeg", "png", "pdf"]
)

engine = st.radio("OCR engine", ["Tesseract", "EasyOCR"], horizontal=True)
lang_hr = st.checkbox("Croatian (HR)", value=True)
lang_en = st.checkbox("English (EN)", value=True)

with st.expander("Pre-processing options"):
    upscale = st.slider("Upscale factor", 1.0, 3.0, 1.5, 0.1)
    do_threshold = st.checkbox("B/W threshold (Otsu)", value=True)
    psm = st.selectbox(
        "Tesseract PSM (page layout mode)",
        [3, 4, 6, 11, 12],
        index=2,
        help="6 = assume uniform block of text (default). 3 = auto. 11 = sparse text."
    )

if uploaded_file:
    name = uploaded_file.name.lower()
    ext = name.split(".")[-1]

    images = []
    if ext == "pdf":
        try:
            pdf_bytes = uploaded_file.read()
            # All pages converted at 300 DPI for good quality
            pages = convert_from_bytes(pdf_bytes, dpi=300)
            images = [p.convert("RGB") for p in pages]
        except Exception as e:
            st.error(f"PDF conversion failed: {e}. Make sure poppler-utils is installed.")
            st.stop()
    else:
        img = Image.open(uploaded_file)
        # Handle transparent PNGs
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        images = [img]

    st.info(f"Pages / images loaded: {len(images)}")

    extracted_all = []

    for idx, img in enumerate(images, start=1):
        st.markdown(f"### Page {idx} of {len(images)}")
        st.image(img, use_container_width=True)

        pre = preprocess_for_ocr(img, upscale=upscale, do_threshold=do_threshold)
        st.image(pre, caption="Pre-processed for OCR", use_container_width=True)

        with st.spinner(f"Running OCR on page {idx}..."):
            text = run_ocr(pre, engine, lang_hr, lang_en, psm)

        extracted_all.append(text)
        st.text_area(f"Extracted text - Page {idx}", value=text, height=220)

    # --- Download full output ---
    final_text = "\n\n".join(
        [f"===== PAGE {i} =====\n{t}" for i, t in enumerate(extracted_all, start=1)]
    )

    st.download_button(
        label="Download all text (.txt)",
        data=final_text.encode("utf-8"),
        file_name="ocr_output.txt",
        mime="text/plain"
    )
