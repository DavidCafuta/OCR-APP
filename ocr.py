import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

from pdf2image import convert_from_bytes

# OCR engines
import pytesseract
import easyocr

st.set_page_config(page_title="OCR HR/EN", page_icon="🔎")
st.title("🔎 OCR (Croatian + English)")
st.markdown("Upload a photo or PDF and extract text (HR/EN).")

def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_pil(cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def preprocess_for_ocr(pil_img, upscale=1.5, do_threshold=True):
    cv_img = pil_to_cv(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if upscale and upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    if do_threshold:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th)
    else:
        return Image.fromarray(gray)

def ocr_tesseract(pil_img, lang_codes="hrv+eng", psm=6):
    # Tesseract language selection and psm config are provided via config string [web:15].
    config = f"-l {lang_codes} --psm {psm}"
    return pytesseract.image_to_string(pil_img, config=config)

@st.cache_resource
def get_easyocr_reader(langs):
    # EasyOCR supports Croatian ('hr') and English ('en') [web:12].
    return easyocr.Reader(langs, gpu=False)

def ocr_easyocr(pil_img, langs=("hr","en")):
    reader = get_easyocr_reader(list(langs))
    # detail=0 returns just text lines
    lines = reader.readtext(np.array(pil_img), detail=0, paragraph=True)
    return "\n".join(lines)

uploaded_file = st.file_uploader("Upload (JPG/PNG/PDF)", type=["jpg","jpeg","png","pdf"])

engine = st.radio("OCR engine", ["Tesseract", "EasyOCR"], horizontal=True)
lang_hr = st.checkbox("Croatian", value=True)
lang_en = st.checkbox("English", value=True)

with st.expander("Pre-processing options"):
    upscale = st.slider("Upscale", 1.0, 3.0, 1.5, 0.1)
    do_threshold = st.checkbox("B/W threshold (Otsu)", value=True)
    psm = st.selectbox("Tesseract PSM (layout mode)", [3,4,6,11,12], index=2)

if uploaded_file:
    name = uploaded_file.name.lower()
    ext = name.split(".")[-1]

    images = []
    if ext == "pdf":
        pdf_bytes = uploaded_file.read()
        # Convert PDF pages to images (DPI tweak for quality).
        pages = convert_from_bytes(pdf_bytes, dpi=300)
        images = [p.convert("RGB") for p in pages]
    else:
        img = Image.open(uploaded_file).convert("RGB")
        images = [img]

    st.write(f"Pages/images loaded: {len(images)}")

    extracted_all = []
    for idx, img in enumerate(images, start=1):
        st.markdown(f"### Page {idx}")
        st.image(img, use_container_width=True)

        pre = preprocess_for_ocr(img, upscale=upscale, do_threshold=do_threshold)
        st.image(pre, caption="Pre-processed for OCR", use_container_width=True)

        langs_selected = []
        if lang_hr: langs_selected.append("hr")
        if lang_en: langs_selected.append("en")
        if not langs_selected:
            st.error("Select at least one language (Croatian and/or English).")
            st.stop()

        if engine == "Tesseract":
            # Tesseract uses ISO-ish traineddata codes; Croatian is typically 'hrv', English 'eng' [web:15].
            tlangs = []
            if lang_hr: tlangs.append("hrv")
            if lang_en: tlangs.append("eng")
            text = ocr_tesseract(pre, lang_codes="+".join(tlangs), psm=psm)
        else:
            text = ocr_easyocr(pre, langs=tuple(langs_selected))

        extracted_all.append(text)

        st.text_area("Extracted text", value=text, height=220)

    final_text = "\n\n-----\n\n".join(extracted_all)
    st.download_button(
        "Download text (.txt)",
        data=final_text.encode("utf-8"),
        file_name="ocr_output.txt",
        mime="text/plain"
    )
