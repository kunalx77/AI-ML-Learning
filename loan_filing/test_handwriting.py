import base64
import json
import os
import time
from io import BytesIO
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Handwritten Loan Form OCR",
    page_icon="📝",
    layout="wide",
)

DEFAULT_MODEL = "gpt-4o"
MAX_RETRIES = 3
PDF_DPI = 250


# ============================================================
# Prompt
# ============================================================

VISION_PROMPT = """
You are an expert document-understanding and handwriting-recognition system.

Analyze this loan application form carefully.

Your task is to extract ONLY the form fields and the handwritten/manual
values filled by the applicant.

Rules:

1. Identify printed form field labels.
2. Associate each handwritten/manual entry with its corresponding field label.
3. Extract handwritten text exactly as written whenever possible.
4. Extract only fields that contain a handwritten/manual value.
5. Ignore printed instructions, headers, footers, logos, boilerplate,
   decorative text, and preprinted examples.
6. Do not copy printed values that are merely part of the form.
7. Do not invent or infer missing values.
8. If a field contains handwriting but the value cannot be determined
   confidently, use null.
9. Preserve dates, phone numbers, addresses, names, and numeric values
   as written.
10. Include checked/selected checkbox fields only when the selected option
    is clearly handwritten/manual or clearly marked by the applicant.
11. Ignore signatures unless the form explicitly labels the signature
    as a field that contains an applicant-entered value.
12. If there are multiple handwritten entries for the same field, choose
    the entry that clearly corresponds to that field.
13. Read the entire page before producing the result.
14. Return ONLY valid JSON.
15. Do not return markdown.
16. Do not return explanations or commentary.

Output format:

{
  "Field Name": "Value",
  "Another Field": "Value"
}

If there are no handwritten/manual fields on the page, return:

{}
"""


# ============================================================
# Helpers
# ============================================================


def get_api_key() -> str:
    """Get API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None

    if key:
        return key

    return os.getenv("OPENAI_API_KEY", "")


def image_to_data_url(image: Image.Image) -> str:
    """
    Convert PIL image to a JPEG data URL suitable for OpenAI image input.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Keep very large scans manageable while retaining handwriting detail.
    max_dimension = 6000

    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
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
        optimize=True,
    )

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Convert every PDF page into a PIL image.
    """
    return convert_from_bytes(
        pdf_bytes,
        dpi=PDF_DPI,
        fmt="jpeg",
        thread_count=2,
    )


def load_uploaded_document(
    uploaded_file,
) -> List[Image.Image]:
    """
    Load JPG/JPEG/PNG directly or convert PDF pages to images.
    """
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return pdf_to_images(file_bytes)

    if filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return [image]

    raise ValueError("Unsupported file type. Please upload PDF, JPG, JPEG, or PNG.")


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Validate and parse the model's JSON response.

    The prompt asks for pure JSON, but this function also handles accidental
    surrounding whitespace or a fenced JSON response defensively.
    """
    if not text:
        raise ValueError("OpenAI returned an empty response.")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("OpenAI response must be a JSON object.")

    # Ensure all values are strings or null.
    normalized = {}

    for field, value in parsed.items():
        if not isinstance(field, str):
            continue

        if value is None:
            normalized[field] = None
        elif isinstance(value, (str, int, float, bool)):
            normalized[field] = str(value)
        else:
            # Do not allow nested/hallucinated structures.
            normalized[field] = None

    return normalized


def call_openai_vision(
    client: OpenAI,
    image: Image.Image,
    model: str,
) -> Dict[str, Any]:
    """
    Send one page to OpenAI Vision with retry handling.
    """
    image_data_url = image_to_data_url(image)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=model,
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
                                "image_url": image_data_url,
                                "detail": "high",
                            },
                        ],
                    }
                ],
                temperature=0,
                max_output_tokens=4000,
            )

            return extract_json_object(response.output_text)

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                # Exponential backoff.
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"OpenAI Vision request failed after {MAX_RETRIES} attempts: " f"{last_error}"
    )


def merge_page_results(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge page-level JSON objects.

    If the same field occurs on multiple pages:
    - Prefer a non-null value.
    - Keep the first value when both are non-null.
    """
    merged = {}

    for page_result in page_results:
        for field, value in page_result.items():
            if field not in merged:
                merged[field] = value
                continue

            current = merged[field]

            if current is None and value is not None:
                merged[field] = value

    return merged


def dataframe_to_json(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert the editable table back into the required JSON structure.
    """
    result = {}

    for _, row in df.iterrows():
        field = str(row["Field Name"]).strip()

        if not field:
            continue

        value = row["Handwritten Value"]

        if pd.isna(value):
            value = None
        else:
            value = str(value).strip()

            if value == "":
                value = None

        result[field] = value

    return result


def json_download_bytes(data: Dict[str, Any]) -> bytes:
    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")


# ============================================================
# UI
# ============================================================

st.title("Handwritten Loan Form OCR")

st.write(
    "Upload a scanned loan application form. The application extracts "
    "only handwritten/manual field values."
)

api_key = get_api_key()

if not api_key:
    st.error(
        "OPENAI_API_KEY is not configured. Set it in your environment "
        "or Streamlit secrets."
    )
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")

    model = st.text_input(
        "Vision model",
        value=DEFAULT_MODEL,
        help="Use a vision-capable OpenAI model available to your API account.",
    )

    st.caption("Each PDF page is analyzed independently using high-detail image input.")

uploaded_file = st.file_uploader(
    "Upload loan application",
    type=["pdf", "jpg", "jpeg", "png"],
    help="Supported formats: PDF, JPG, JPEG, PNG",
)


# ============================================================
# Process document
# ============================================================

if uploaded_file is not None:
    st.divider()

    try:
        with st.spinner("Reading document..."):
            pages = load_uploaded_document(uploaded_file)

    except Exception as exc:
        st.error(f"Unable to read the uploaded file: {exc}")
        st.stop()

    if not pages:
        st.error("The document contains no readable pages.")
        st.stop()

    st.success(f"Document loaded successfully: {len(pages)} page(s)")

    with st.expander("Preview uploaded document"):
        for page_number, page in enumerate(pages, start=1):
            st.image(
                page,
                caption=f"Page {page_number}",
                use_container_width=True,
            )

    process_button = st.button(
        "Extract Handwritten Fields",
        type="primary",
        use_container_width=True,
    )

    if process_button:
        page_results = []

        progress = st.progress(0)

        status = st.empty()

        for index, page in enumerate(pages):
            page_number = index + 1

            status.info(f"Analyzing page {page_number} of {len(pages)}...")

            try:
                result = call_openai_vision(
                    client=client,
                    image=page,
                    model=model,
                )

                page_results.append(result)

            except Exception as exc:
                status.error(f"Could not process page {page_number}: {exc}")

                # Continue processing remaining pages.
                page_results.append({})

            progress.progress(page_number / len(pages))

        status.empty()

        final_result = merge_page_results(page_results)

        # Save in session state so table edits survive reruns.
        st.session_state["ocr_result"] = final_result

        if not final_result:
            st.warning("No handwritten fields were detected in the uploaded document.")


# ============================================================
# Results
# ============================================================

if "ocr_result" in st.session_state:
    result = st.session_state["ocr_result"]

    if result:
        st.divider()
        st.subheader("Extracted Fields")

        tab_json, tab_table = st.tabs(["JSON Viewer", "Editable Table"])

        # ----------------------------------------------------
        # JSON viewer
        # ----------------------------------------------------

        with tab_json:
            st.json(result)

            download_data = json_download_bytes(result)

            st.download_button(
                label="Download JSON",
                data=download_data,
                file_name="handwritten_loan_fields.json",
                mime="application/json",
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Editable table
        # ----------------------------------------------------

        with tab_table:
            rows = [
                {
                    "Field Name": field,
                    "Handwritten Value": value,
                }
                for field, value in result.items()
            ]

            df = pd.DataFrame(rows)

            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Field Name": st.column_config.TextColumn(
                        "Field Name",
                        required=True,
                    ),
                    "Handwritten Value": st.column_config.TextColumn(
                        "Handwritten Value",
                    ),
                },
                key="ocr_editor",
            )

            if st.button(
                "Update JSON From Table",
                use_container_width=True,
            ):
                updated_result = dataframe_to_json(edited_df)

                st.session_state["ocr_result"] = updated_result

                st.success("JSON updated from the table.")

                st.rerun()

            st.download_button(
                label="Download Edited JSON",
                data=json_download_bytes(dataframe_to_json(edited_df)),
                file_name="handwritten_loan_fields.json",
                mime="application/json",
                use_container_width=True,
            )
