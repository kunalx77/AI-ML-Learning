import io
import os
import base64

import cv2
import numpy as np
import pytesseract

from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pdf2image import convert_from_bytes, convert_from_path
from openai import OpenAI

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY was not found in the .env file.")

OPENAI_VISION_MODEL = os.getenv(
    "OPENAI_VISION_MODEL",
    "gpt-4.1",
)

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# PDF SETTINGS
# ============================================================

# Higher DPI gives the vision model more readable text.
PDF_DPI = int(
    os.getenv(
        "PDF_DPI",
        "250",
    )
)


# ============================================================
# CHECKBOX DETECTION
# ============================================================


def detect_checkboxes(image):

    image_array = np.array(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY,
    )

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    checkboxes = []

    for contour in contours:

        x, y, width, height = cv2.boundingRect(contour)

        if width < 10 or height < 10:
            continue

        if width > 50 or height > 50:
            continue

        ratio = width / float(height)

        if not 0.7 <= ratio <= 1.3:
            continue

        area = cv2.contourArea(contour)

        if area < 40:
            continue

        if area > width * height * 0.9:
            continue

        checkboxes.append(
            (
                x,
                y,
                width,
                height,
            )
        )

    checkboxes.sort(
        key=lambda item: (
            item[1],
            item[0],
        )
    )

    return checkboxes


# ============================================================
# TEST CHECKBOX DETECTION
# ============================================================


def test_checkbox_detection(image):

    checkboxes = detect_checkboxes(image)

    print("\ncheckboxes detected:")

    for index, checkbox in enumerate(
        checkboxes,
        start=1,
    ):

        x, y, width, height = checkbox

        print(
            f"checkbox {index}: "
            f"x={x}, "
            f"y={y}, "
            f"width={width}, "
            f"height={height}"
        )

    return checkboxes


# ============================================================
# IMAGE PREPROCESSING
# ============================================================


def preprocess_image(image):

    image = ImageOps.grayscale(image)

    width, height = image.size

    scale = 2

    image = image.resize(
        (
            width * scale,
            height * scale,
        ),
        Image.Resampling.LANCZOS,
    )

    image = ImageEnhance.Contrast(image).enhance(1.8)

    image = image.filter(ImageFilter.SHARPEN)

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=2,
            percent=130,
            threshold=3,
        )
    )

    return image


# ============================================================
# LOCAL OCR
# ============================================================


def extract_page_text(image):

    processed_image = preprocess_image(image)

    results = []

    try:

        layout_text = pytesseract.image_to_string(
            processed_image,
            config=("--psm 6 " "-c preserve_interword_spaces=1"),
        )

        results.append("LAYOUT OCR:\n" + layout_text)

    except Exception:
        pass

    try:

        normal_text = pytesseract.image_to_string(
            processed_image,
            config="--psm 6",
        )

        results.append("NORMAL OCR:\n" + normal_text)

    except Exception:
        pass

    try:

        sparse_text = pytesseract.image_to_string(
            processed_image,
            config="--psm 11",
        )

        results.append("SPARSE OCR:\n" + sparse_text)

    except Exception:
        pass

    return "\n\n".join(results)


# ============================================================
# IMAGE TO BASE64
# ============================================================


def image_to_data_url(image):

    buffer = io.BytesIO()

    # Always send RGB.
    image = image.convert("RGB")

    image.save(
        buffer,
        format="JPEG",
        quality=95,
        optimize=True,
    )

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return "data:image/jpeg;base64," + encoded


# ============================================================
# OPENAI VISION
# ============================================================


def openai_vision_page(
    image,
    page_number,
):

    image_data_url = image_to_data_url(image)

    prompt = f"""
You are performing high-accuracy OCR on page
{page_number} of a loan application document.

IMPORTANT:

Analyze the ENTIRE IMAGE.

Do not only read the top half.

Do not stop after finding the first fields.

Read:

- top of page
- middle of page
- bottom of page
- left side
- right side
- tables
- small text
- checkboxes
- handwritten information
- printed information
- signatures
- fields near page edges

Return a COMPLETE transcription of everything
meaningful and readable on this page.

Preserve the document's approximate reading order.

Preserve:

- names
- dates
- phone numbers
- addresses
- email addresses
- amounts
- identification numbers
- application numbers
- account numbers
- field labels
- checkbox labels
- selected checkbox states
- table values
- signature indicators

Do NOT invent information.

If text cannot be read confidently, indicate:

[UNREADABLE]

instead of guessing.

CHECKBOXES:

If a checkbox is visibly selected, write:

[CHECKED] Label

If it is visibly unselected, write:

[UNCHECKED] Label

If the state cannot be determined, write:

[UNKNOWN] Label

IMPORTANT:

Do not omit any field merely because it is blank.

Blank fields should still be represented by their
visible labels when possible.

Return plain text only.

Do not summarize.

Do not explain.

Do not provide JSON.

Start exactly with:

PAGE: {page_number}
"""

    response = client.responses.create(
        model=OPENAI_VISION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    text = response.output_text

    if not text or not text.strip():
        raise Exception(f"OpenAI returned empty OCR for page {page_number}.")

    return text.strip()


# ============================================================
# PROCESS ALL PAGES
# ============================================================


def perform_openai_ocr(pages):

    all_text = []

    total_pages = len(pages)

    if total_pages == 0:
        raise Exception("No pages were found in the document.")

    print(f"Detected {total_pages} page(s).")

    for page_number, page in enumerate(
        pages,
        start=1,
    ):

        print(f"Processing page " f"{page_number}/{total_pages}...")

        try:

            # ------------------------------------------------
            # Ensure the page is fully loaded.
            # ------------------------------------------------

            page.load()

            # ------------------------------------------------
            # Convert to RGB.
            # ------------------------------------------------

            page = page.convert("RGB")

            # ------------------------------------------------
            # Send this page independently.
            # ------------------------------------------------

            page_text = openai_vision_page(
                page,
                page_number,
            )

            all_text.append(f"\n===== PAGE {page_number} =====\n\n" f"{page_text}")

            print(f"Page {page_number} completed.")

        except Exception as e:

            print(f"OpenAI Vision failed on " f"page {page_number}: {e}")

            # ------------------------------------------------
            # Do NOT silently lose this page.
            # ------------------------------------------------

            try:

                fallback_text = extract_page_text(page)

                if fallback_text.strip():

                    all_text.append(
                        f"\n===== PAGE {page_number} "
                        f"(LOCAL OCR FALLBACK) =====\n\n"
                        f"{fallback_text}"
                    )

                    print(f"Page {page_number} " "completed using local OCR fallback.")

                else:

                    all_text.append(
                        f"\n===== PAGE {page_number} "
                        "OCR FAILED =====\n\n"
                        "[OCR FAILED FOR THIS PAGE]"
                    )

            except Exception as fallback_error:

                all_text.append(
                    f"\n===== PAGE {page_number} "
                    "OCR FAILED =====\n\n"
                    "[OCR FAILED FOR THIS PAGE]\n"
                    f"Error: {fallback_error}"
                )

    # ========================================================
    # VERIFY PAGE COUNT
    # ========================================================

    result = "\n".join(all_text)

    processed_count = result.count("===== PAGE ")

    print(f"Finished processing " f"{processed_count}/{total_pages} pages.")

    return result


# ============================================================
# BACKEND FORM OCR
# ============================================================


def get_ocr(pdf_path):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = convert_from_path(
        pdf_path,
        dpi=PDF_DPI,
        fmt="png",
        thread_count=1,
    )

    if not pages:
        raise Exception("PDF was opened but no pages were rendered.")

    return perform_openai_ocr(pages)


# ============================================================
# UPLOADED FILE OCR
# ============================================================


def perform_ocr(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise Exception("Uploaded file is empty.")

    file_name = uploaded_file.name.lower()

    # ========================================================
    # PDF
    # ========================================================

    if file_name.endswith(".pdf"):

        pages = convert_from_bytes(
            file_bytes,
            dpi=PDF_DPI,
            fmt="png",
            thread_count=1,
        )

        if not pages:
            raise Exception("The PDF was opened but no pages " "could be rendered.")

    # ========================================================
    # IMAGE
    # ========================================================

    else:

        try:

            image = Image.open(io.BytesIO(file_bytes))

            image.load()

            image = image.convert("RGB")

            pages = [image]

        except Exception as e:

            raise Exception(f"Unable to open uploaded image: {e}")

    # ========================================================
    # PROCESS EVERY PAGE
    # ========================================================

    return perform_openai_ocr(pages)


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    pdf_path = "forms/home_loan.pdf"

    print("\n" + "=" * 70)

    print("PDF TEST")

    print("=" * 70)

    pages = convert_from_path(
        pdf_path,
        dpi=PDF_DPI,
        fmt="png",
        thread_count=1,
    )

    print(f"PDF contains " f"{len(pages)} page(s).")

    # --------------------------------------------------------
    # Checkbox testing
    # --------------------------------------------------------

    for page_number, page in enumerate(
        pages,
        start=1,
    ):

        print("\n" + "=" * 60)

        print(f"PAGE {page_number} " "CHECKBOX DETECTION")

        print("=" * 60)

        test_checkbox_detection(page)

    # --------------------------------------------------------
    # Vision OCR testing
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("RUNNING OPENAI VISION OCR")

    print("=" * 70)

    try:

        result = perform_openai_ocr(pages)

        print("\n" + result)

    except Exception as e:

        print("\nOCR ERROR:")

        print(e)
