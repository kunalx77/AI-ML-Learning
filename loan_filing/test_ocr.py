import io

import pytesseract


import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pdf2image import convert_from_bytes, convert_from_path

# detect checkbox regions in an image


def detect_checkboxes(image):

    image_array = np.array(image)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

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

        if 0.7 <= ratio <= 1.3:

            area = cv2.contourArea(contour)

            if area < 40:
                continue

            if area > width * height * 0.9:
                continue

            checkboxes.append((x, y, width, height))

    checkboxes.sort(key=lambda item: (item[1], item[0]))

    return checkboxes


# test checkbox detection


def test_checkbox_detection(image):

    checkboxes = detect_checkboxes(image)

    print("\ncheckboxes detected:")

    for index, checkbox in enumerate(checkboxes, start=1):

        x, y, width, height = checkbox

        print(f"checkbox {index}: " f"x={x}, y={y}, width={width}, height={height}")

    return checkboxes


# preprocess image for better ocr


def preprocess_image(image):

    image = ImageOps.grayscale(image)

    width, height = image.size

    scale = 2

    image = image.resize((width * scale, height * scale), Image.Resampling.LANCZOS)

    image = ImageEnhance.Contrast(image).enhance(1.8)

    image = image.filter(ImageFilter.SHARPEN)

    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=3))

    return image


# perform multiple ocr passes


def extract_page_text(image):

    processed_image = preprocess_image(image)

    results = []

    # layout preserving ocr

    try:

        layout_text = pytesseract.image_to_string(
            processed_image, config="--psm 6 -c preserve_interword_spaces=1"
        )

        results.append("LAYOUT OCR:\n" + layout_text)

    except Exception:

        pass

    # normal block ocr

    try:

        normal_text = pytesseract.image_to_string(processed_image, config="--psm 6")

        results.append("NORMAL OCR:\n" + normal_text)

    except Exception:

        pass

    # sparse text ocr

    try:

        sparse_text = pytesseract.image_to_string(processed_image, config="--psm 11")

        results.append("SPARSE OCR:\n" + sparse_text)

    except Exception:

        pass

    return "\n\n".join(results)


# get ocr from backend form


def get_ocr(pdf_path):

    pages = convert_from_path(pdf_path, dpi=200)

    all_text = []

    for page_number, page in enumerate(pages, start=1):

        page_text = extract_page_text(page)

        all_text.append(f"\n===== PAGE {page_number} =====\n\n" f"{page_text}")

    return "\n".join(all_text)


# perform ocr on uploaded pdf or image


def perform_ocr(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    file_name = uploaded_file.name.lower()

    all_text = []

    # pdf

    if file_name.endswith(".pdf"):

        pages = convert_from_bytes(file_bytes, dpi=200)

        for page_number, page in enumerate(pages, start=1):

            page_text = extract_page_text(page)

            all_text.append(f"\n===== PAGE {page_number} =====\n\n" f"{page_text}")

    # image

    else:

        image = Image.open(io.BytesIO(file_bytes))

        page_text = extract_page_text(image)

        all_text.append(f"\n===== PAGE 1 =====\n\n" f"{page_text}")

    return "\n".join(all_text)


if __name__ == "__main__":

    pdf_path = "forms/home_loan.pdf"

    pages = convert_from_path(pdf_path, dpi=180)

    for page_number, page in enumerate(pages, start=1):

        print("\n" + "=" * 60)
        print(f"page {page_number} checkbox detection")
        print("=" * 60)

        test_checkbox_detection(page)
