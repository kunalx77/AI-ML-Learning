import base64
import json
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes
from openai import OpenAI

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Handwritten Loan Form OCR",
    page_icon="📝",
    layout="wide",
)

st.title("Handwritten Loan Form OCR")
st.caption("Upload a scanned loan application and extract handwritten " "field values.")


# ============================================================
# OPENAI
# ============================================================

api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key was not found.")
    st.stop()

client = OpenAI(api_key=api_key)


# ============================================================
# PROMPT
# ============================================================

VISION_PROMPT = """
You are an expert document-understanding and handwriting-recognition
system.

Analyze the ENTIRE page carefully.

Extract ONLY form fields that have been manually filled by the applicant.

For every handwritten/manual entry:

- Identify the printed field label.
- Read the handwritten value associated with that field.
- Return the field label and handwritten value.
- Preserve the value exactly as written whenever possible.
- Preserve spelling, numbers, dates, phone numbers and addresses.
- Do not correct spelling.
- Do not infer missing information.
- Do not hallucinate.

Ignore:

- Printed instructions
- Printed examples
- Headers
- Footers
- Logos
- Decorative text
- Boilerplate text
- Pre-filled printed information

If a handwritten field exists but its value cannot be read confidently,
return null for that value.

Ignore signatures unless the signature is explicitly a normal form field.

Return ONLY valid JSON.

Example:

{
    "Applicant Name": "Rahul Patel",
    "Date of Birth": "12/05/1994",
    "Mobile Number": "9876543210",
    "Address": "Ahmedabad, Gujarat"
}

If this page has no handwritten fields, return:

{}
"""


# ============================================================
# IMAGE → DATA URL
# ============================================================


def image_to_data_url(image):

    image = ImageOps.exif_transpose(image).convert("RGB")

    # Keep enough resolution for handwriting.
    max_size = 6000

    if max(image.size) > max_size:

        scale = max_size / max(image.size)

        image = image.resize(
            (
                int(image.width * scale),
                int(image.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# VISION EXTRACTION
# ============================================================


def extract_page(image):

    image_url = image_to_data_url(image)

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": VISION_PROMPT,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    text = response.output_text.strip()

    # Defensive handling of accidental markdown.
    if text.startswith("```"):

        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    result = json.loads(text)

    if not isinstance(result, dict):
        raise ValueError("The model did not return a JSON object.")

    return result


# ============================================================
# MERGE RESULTS FROM ALL PAGES
# ============================================================


def merge_results(page_results):

    final_result = {}

    for page_result in page_results:

        for field, value in page_result.items():

            field = str(field).strip()

            if not field:
                continue

            # If field doesn't exist, add it.
            if field not in final_result:

                final_result[field] = value

            # If previous value was null but current value isn't,
            # use the current value.
            elif final_result[field] is None and value is not None:

                final_result[field] = value

    return final_result


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload loan application",
    type=[
        "pdf",
        "jpg",
        "jpeg",
        "png",
    ],
)


if uploaded_file:

    filename = uploaded_file.name.lower()

    # ========================================================
    # LOAD DOCUMENT
    # ========================================================

    try:

        if filename.endswith(".pdf"):

            pdf_bytes = uploaded_file.getvalue()

            pages = convert_from_bytes(
                pdf_bytes,
                dpi=250,
            )

        else:

            image = Image.open(BytesIO(uploaded_file.getvalue()))

            pages = [image]

    except Exception as e:

        st.error(f"Could not read the document: {e}")

        st.stop()

    st.success(f"Document loaded — {len(pages)} page(s)")

    # ========================================================
    # PROCESS BUTTON
    # ========================================================

    if st.button(
        "Read Handwritten Fields",
        type="primary",
        use_container_width=True,
    ):

        all_results = []

        progress = st.progress(0)

        status = st.empty()

        # ====================================================
        # PROCESS EVERY PAGE
        # ====================================================

        for index, page in enumerate(pages):

            page_number = index + 1

            status.write(f"Reading page {page_number} of {len(pages)}...")

            try:

                result = extract_page(page)

                all_results.append(result)

            except Exception as e:

                st.warning(f"Could not process page {page_number}: {e}")

                all_results.append({})

            progress.progress(page_number / len(pages))

        status.empty()

        progress.empty()

        # ====================================================
        # MERGE
        # ====================================================

        final_result = merge_results(all_results)

        # ====================================================
        # DISPLAY
        # ====================================================

        st.divider()

        st.subheader("Extracted Handwritten Information")

        if not final_result:

            st.warning("No handwritten fields were detected.")

        else:

            rows = []

            for field, value in final_result.items():

                rows.append(
                    {
                        "Field": field,
                        "Handwritten Value": ("" if value is None else str(value)),
                    }
                )

            df = pd.DataFrame(rows)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Field": st.column_config.TextColumn(
                        "Field",
                        width="medium",
                    ),
                    "Handwritten Value": st.column_config.TextColumn(
                        "Handwritten Value",
                        width="large",
                    ),
                },
            )

            st.success(f"{len(final_result)} handwritten field(s) extracted.")
