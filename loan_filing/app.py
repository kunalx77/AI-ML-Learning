import os
import io
import base64

import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path

from form_extractor import extract_form_fields

from id_extractor import (
    extract_id_information,
    parse_id_information,
    match_id_to_form,
    parse_matched_fields,
    extract_filled_form_information,
)

# ============================================================
# ENVIRONMENT / OPENAI CONFIGURATION
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VISION_MODEL = os.getenv(
    "OPENAI_VISION_MODEL",
    "gpt-5.4-mini",
)


def create_openai_client():
    """
    Create the OpenAI client using the environment variable.

    No API key is hardcoded in the application.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY was not found. "
            "Please add OPENAI_API_KEY to your .env file."
        )

    return OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# OPENAI VISION OCR
# ============================================================


def image_to_data_url(image):
    """
    Convert a PIL image into a base64 JPEG data URL
    suitable for OpenAI vision input.
    """

    buffer = io.BytesIO()

    # Convert to RGB so PNG/RGBA/grayscale images are safe to encode as JPEG.
    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded_image}"


def extract_text_from_image_with_openai(image):
    """
    Send one document page/image to OpenAI vision and return
    OCR-style text.

    Important:
    This function only performs document text extraction.

    It does NOT:
    - identify loan fields
    - validate fields
    - match ID information
    - modify application logic

    Those existing responsibilities remain in the existing
    extractor modules.
    """

    client = create_openai_client()

    image_data_url = image_to_data_url(image)

    prompt = """
You are a high-accuracy document OCR engine.

Read the ENTIRE document page shown in the image.

Extract all visible text from the document.

IMPORTANT RULES:

1. Preserve the original reading order as closely as possible.

2. Preserve form labels exactly or as closely as possible.

3. Preserve values exactly as they appear.

4. Preserve numbers, dates, phone numbers, email addresses,
   identification numbers, currency amounts, names and addresses.

5. Preserve checkbox and selection labels.

6. If a checkbox is visibly checked or selected, represent it
   clearly, for example:

   [X] New Vehicle
   [ ] Used Vehicle

7. If a checkbox is visibly unchecked, represent it as:

   [ ] Option Name

8. Preserve table rows and columns as reasonably as possible
   using spaces or lines.

9. Do not summarize the document.

10. Do not interpret the document.

11. Do not guess missing text.

12. Do not invent values.

13. Do not add explanations.

14. Do not convert the document into JSON.

15. Do not extract only important information.

16. Extract ALL readable text from the entire page.

17. If text is unclear, make the best OCR transcription possible
    without inventing information.

18. Keep headings and section names.

19. Keep field labels separate from their values when the layout
    makes that distinction clear.

20. Maintain approximate spatial/layout relationships whenever
    possible.

Return ONLY the OCR text.
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

    return response.output_text


def get_ocr(pdf_path):
    """
    OCR an existing backend PDF form using OpenAI vision.

    This keeps the same function name used by the existing
    application, so the rest of the application does not need
    to know that the OCR engine has changed.
    """

    pages = convert_from_path(
        pdf_path,
        dpi=200,
    )

    all_text = []

    for page_number, page in enumerate(
        pages,
        start=1,
    ):
        page_text = extract_text_from_image_with_openai(page)

        all_text.append(f"\n===== PAGE {page_number} =====\n\n" f"{page_text}")

    return "\n".join(all_text)


def perform_ocr(uploaded_file):
    """
    OCR an uploaded PDF or image using OpenAI vision.

    The return value remains a plain OCR text string so the
    existing form/id extraction code can continue working
    unchanged.
    """

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    all_text = []

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if file_name.endswith(".pdf"):

        pages = convert_from_bytes(
            file_bytes,
            dpi=200,
        )

        for page_number, page in enumerate(
            pages,
            start=1,
        ):
            page_text = extract_text_from_image_with_openai(page)

            all_text.append(f"\n===== PAGE {page_number} =====\n\n" f"{page_text}")

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    else:

        image = Image.open(io.BytesIO(file_bytes))

        page_text = extract_text_from_image_with_openai(image)

        all_text.append("\n===== PAGE 1 =====\n\n" f"{page_text}")

    return "\n".join(all_text)


# ============================================================
# CONFIGURATION
# ============================================================

FORMS_FOLDER = "forms"

st.set_page_config(
    page_title="Loan Assistant",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOAN MAPPING
# ============================================================

loan_map = {
    "Business Loan": (
        "business_loan.pdf",
        "💼",
    ),
    "Car Loan": (
        "car_loan.pdf",
        "🚗",
    ),
    "Education Loan": (
        "education_loan.pdf",
        "🎓",
    ),
    "Home Loan": (
        "home_loan.pdf",
        "🏠",
    ),
    "Personal Loan": (
        "personal_loan.pdf",
        "👤",
    ),
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       APP
    ===================================================== */

    .stApp {
        background-color: #f5f7fb;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 24px;
        padding-bottom: 90px;
    }

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #172033 !important;
    }

    p {
        color: #475569;
    }


    /* =====================================================
       HEADER
    ===================================================== */

    .header-title {
        font-size: 25px;
        font-weight: 750;
        color: #172033 !important;
        margin-bottom: 2px;
    }

    .header-subtitle {
        font-size: 13px;
        color: #64748b !important;
    }

    .online-box {
        color: #15803d !important;
        background-color: #f0fdf4 !important;
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 6px 12px;
        text-align: center;
        font-size: 12px;
        font-weight: 600;
    }


    /* =====================================================
       PROGRESS
    ===================================================== */

    .progress-container {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 8px;
        margin-bottom: 20px;
    }

    .progress-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b !important;
        margin-bottom: 9px;
    }


    /* =====================================================
       PROCESSING PROGRESS
    ===================================================== */

    .processing-stage {
        font-size: 13px;
        color: #334155 !important;
        font-weight: 600;
        margin-bottom: 5px;
    }


    /* =====================================================
       ASSISTANT BUBBLE
    ===================================================== */

    .assistant-bubble {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 5px 16px 16px 16px;
        padding: 13px 16px;
        margin: 7px 0 13px 0;
        color: #172033 !important;
        line-height: 1.55;
        box-shadow: 0 2px 7px rgba(15, 23, 42, 0.035);
    }

    .assistant-bubble * {
        color: #172033 !important;
    }


    /* =====================================================
       USER BUBBLE
    ===================================================== */

    .user-bubble {
        background-color: #1d4ed8 !important;
        border: 1px solid #1e40af !important;
        border-radius: 16px 5px 16px 16px;
        padding: 13px 16px;
        margin: 7px 0 13px auto;
        color: #ffffff !important;
        line-height: 1.55;
        max-width: 80%;
        font-weight: 500;
        box-shadow: 0 2px 7px rgba(30, 64, 175, 0.18);
    }

    .user-bubble,
    .user-bubble *,
    .user-bubble span,
    .user-bubble p,
    .user-bubble strong,
    .user-bubble b {
        color: #ffffff !important;
    }


    /* =====================================================
       MESSAGE LABEL
    ===================================================== */

    .message-label {
        font-size: 11px;
        font-weight: 700;
        color: #64748b !important;
        margin-bottom: 3px;
    }

    .user-label {
        text-align: right;
    }


    /* =====================================================
       CARDS
    ===================================================== */

    .info-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px;
        margin: 10px 0;
        color: #172033 !important;
    }

    .info-card * {
        color: #172033 !important;
    }

    .info-card-title {
        color: #334155 !important;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        padding: 8px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 13px;
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-key {
        color: #64748b !important;
    }

    .info-value {
        color: #172033 !important;
        font-weight: 600;
        text-align: right;
        word-break: break-word;
    }


    /* =====================================================
       SYSTEM MESSAGE
    ===================================================== */

    .system-box {
        background-color: #f8fafc !important;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 12px 14px;
        margin: 10px 0;
        color: #475569 !important;
        font-size: 13px;
    }

    .system-box * {
        color: #475569 !important;
    }


    /* =====================================================
       SUCCESS
    ===================================================== */

    .success-box {
        background-color: #f0fdf4 !important;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 13px 15px;
        margin: 10px 0;
        color: #166534 !important;
    }

    .success-box * {
        color: #166534 !important;
    }


    /* =====================================================
       WARNING
    ===================================================== */

    .warning-box {
        background-color: #fffbeb !important;
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 13px 15px;
        margin: 10px 0;
        color: #92400e !important;
    }


    /* =====================================================
       ERROR
    ===================================================== */

    .error-box {
        background-color: #fef2f2 !important;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 13px 15px;
        margin: 10px 0;
        color: #991b1b !important;
    }


    /* =====================================================
       BUTTONS
    ===================================================== */

    div.stButton > button {
        min-height: 43px;
        border-radius: 10px;
        border: 1px solid #d7dee8;
        background-color: #ffffff !important;
        color: #172033 !important;
        font-weight: 600;
    }

    div.stButton > button:hover {
        border-color: #2563eb !important;
        color: #1d4ed8 !important;
        background-color: #eff6ff !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }

    div.stButton > button[kind="primary"] * {
        color: #ffffff !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        border-color: #1d4ed8 !important;
    }


    /* =====================================================
       TEXT INPUT
    ===================================================== */

    .stTextInput input {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-color: #cbd5e1 !important;
        border-radius: 10px !important;
    }

    .stTextInput label {
        color: #334155 !important;
        font-weight: 600 !important;
    }


    /* =====================================================
       SELECTBOX
    ===================================================== */

    .stSelectbox label {
        color: #334155 !important;
        font-weight: 600 !important;
    }


    /* =====================================================
       FILE UPLOADER
    ===================================================== */

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
    }

    [data-testid="stFileUploader"] * {
        color: #334155;
    }


    /* =====================================================
       CHAT INPUT
    ===================================================== */

    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #172033 !important;
    }


    /* =====================================================
       STREAMLIT CHROME
    ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "step": 0,
    "loan_type": None,
    "filling_method": None,
    "form_sections": None,
    "filled_form_sections": None,
    "filled_form_values": None,
    "id_form_sections": None,
    "matched_fields": None,
    "final_form_values": {},
    "submitted": False,
    "messages": [],
    "conversation_initialized": False,
    "current_field_index": 0,
    "current_field_section": None,
    "current_field": None,
    "document_processed": False,
    "extraction_confirmed": False,
    "chat_error": None,
}


for key, value in defaults.items():

    if key not in st.session_state:

        if isinstance(value, dict):

            st.session_state[key] = {}

        elif isinstance(value, list):

            st.session_state[key] = []

        else:

            st.session_state[key] = value


# ============================================================
# FORM PARSER
# ============================================================


def parse_form_fields(extraction):

    sections = []

    current_section = None

    current_fields = []

    unsectioned_fields = []

    for line in extraction.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("SECTION:"):

            if current_section is None and unsectioned_fields:

                sections.append(
                    (
                        "Loan Requested",
                        unsectioned_fields,
                    )
                )

                unsectioned_fields = []

            elif current_section:

                sections.append(
                    (
                        current_section,
                        current_fields,
                    )
                )

            current_section = line[len("SECTION:") :].strip()

            current_fields = []

        elif line.startswith("FIELD:"):

            field_name = line[len("FIELD:") :].strip()

            if current_section:

                current_fields.append(field_name)

            else:

                unsectioned_fields.append(field_name)

    if current_section:

        sections.append(
            (
                current_section,
                current_fields,
            )
        )

    if unsectioned_fields:

        sections.insert(
            0,
            (
                "Loan Requested",
                unsectioned_fields,
            ),
        )

    return sections


# ============================================================
# CHAT HELPERS
# ============================================================


def add_message(role, content):

    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
        }
    )


def initialize_conversation():

    if not st.session_state.conversation_initialized:

        add_message(
            "assistant",
            "Hi! 👋 Welcome to the Loan Assistant. "
            "I'll guide you through your loan application "
            "step by step. Let's start by choosing your loan type.",
        )

        st.session_state.conversation_initialized = True


def clean_value(value):

    if value is None:
        return ""

    value = str(value).strip()

    if value == "Empty/Not Filled":
        return ""

    return value


def get_sections():

    if st.session_state.form_sections:
        return st.session_state.form_sections

    if st.session_state.filled_form_sections:
        return st.session_state.filled_form_sections

    if st.session_state.id_form_sections:
        return st.session_state.id_form_sections

    return []


def get_all_fields():

    fields = []

    for section, section_fields in get_sections():

        for field in section_fields:

            fields.append(
                (
                    section,
                    field,
                )
            )

    return fields


def get_application_value(field):

    return clean_value(
        st.session_state.final_form_values.get(
            field,
            "",
        )
    )


def set_application_value(field, value):

    if "final_form_values" not in st.session_state:

        st.session_state.final_form_values = {}

    st.session_state.final_form_values[field] = value


def update_current_field():

    fields = get_all_fields()

    for index, (section, field) in enumerate(fields):

        if not get_application_value(field):

            st.session_state.current_field_index = index

            st.session_state.current_field_section = section

            st.session_state.current_field = field

            return section, field

    st.session_state.current_field_index = len(fields)

    st.session_state.current_field_section = None

    st.session_state.current_field = None

    return None, None


# ============================================================
# PROCESSING STATUS
# ============================================================


def create_processing_ui(title):

    status = st.status(
        title,
        expanded=True,
    )

    progress = st.progress(
        0,
        text="Uploading Document...",
    )

    return status, progress


def update_processing_stage(
    status,
    progress,
    percentage,
    message,
):

    progress.progress(
        percentage,
        text=message,
    )

    status.write(message)


def finish_processing_ui(
    status,
    progress,
):

    progress.progress(
        100,
        text="Completed ✓",
    )

    status.update(
        label="Completed ✓",
        state="complete",
        expanded=False,
    )


# ============================================================
# HEADER
# ============================================================


def render_header():

    left, right = st.columns(
        [5, 1],
        vertical_alignment="center",
    )

    with left:

        st.markdown(
            '<div class="header-title">' "🏦 Loan Assistant" "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="header-subtitle">'
            "Secure loan application assistant"
            "</div>",
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            '<div class="online-box">' "● Online" "</div>",
            unsafe_allow_html=True,
        )

    st.divider()


# ============================================================
# PROGRESS
# ============================================================


def get_progress_index():

    step = st.session_state.step

    if step in (0, 1):
        return 0

    if step == 2:
        return 1

    if step == 3:
        return 3

    if step == 4:
        return 4

    return 0


def render_progress():

    stages = [
        "Loan",
        "Information",
        "Documents",
        "Review",
        "Submitted",
    ]

    current = get_progress_index()

    st.markdown(
        '<div class="progress-container">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="progress-title">' "Application Progress" "</div>",
        unsafe_allow_html=True,
    )

    parts = []

    for index, stage in enumerate(stages):

        if index < current:

            parts.append(
                f'<span style="color:#15803d;'
                f'font-weight:600;">'
                f"✓ {stage}"
                f"</span>"
            )

        elif index == current:

            parts.append(
                f'<span style="color:#2563eb;'
                f'font-weight:700;">'
                f"● {stage}"
                f"</span>"
            )

        else:

            parts.append(f'<span style="color:#94a3b8;">' f"{stage}" f"</span>")

    st.markdown(
        " &nbsp; → &nbsp; ".join(parts),
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT HISTORY
# ============================================================


def render_chat():

    for message in st.session_state.messages:

        role = message["role"]

        content = message["content"]

        if role == "user":

            st.markdown(
                '<div class="message-label user-label">' "You" "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="user-bubble">' f"{content}" f"</div>",
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="message-label">' "Loan Assistant" "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="assistant-bubble">' f"{content}" f"</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# LOAN SELECTION
# ============================================================


def render_loan_selection():

    st.markdown(
        '<div class="system-box">'
        "Choose the loan you would like to apply for."
        "</div>",
        unsafe_allow_html=True,
    )

    for loan_name, (_, icon) in loan_map.items():

        if st.button(
            f"{icon}  {loan_name}",
            key=f"loan_{loan_name}",
            use_container_width=True,
        ):

            st.session_state.loan_type = loan_name

            st.session_state.step = 1

            add_message(
                "user",
                f"{icon} {loan_name}",
            )

            add_message(
                "assistant",
                f"Great. You're applying for a "
                f"**{loan_name}**. "
                f"How would you like to provide your information?",
            )

            st.rerun()


# ============================================================
# FILLING METHOD
# ============================================================


def render_filling_method():

    loan_name = st.session_state.loan_type

    if loan_name:

        icon = loan_map[loan_name][1]

        st.markdown(
            """
            <div style="
                color:#64748b;
                font-size:12px;
                font-weight:700;
                margin-top:8px;
                margin-bottom:5px;
            ">
                SELECTED LOAN
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success(f"{icon}  {loan_name}")

    st.markdown(
        '<div class="system-box">'
        "Choose how you would like to provide your "
        "application information."
        "</div>",
        unsafe_allow_html=True,
    )

    options = [
        (
            "✍️ Fill application manually",
            "Fill Manually",
        ),
        (
            "📄 Upload a filled application",
            "Upload Filled Form",
        ),
        (
            "🪪 Upload a government ID",
            "Upload Government ID",
        ),
    ]

    for label, method in options:

        if st.button(
            label,
            key=f"method_{method}",
            use_container_width=True,
        ):

            st.session_state.filling_method = method

            st.session_state.step = 2

            add_message(
                "user",
                label,
            )

            if method == "Fill Manually":

                add_message(
                    "assistant",
                    "Perfect. I'll ask for your application "
                    "information one field at a time.",
                )

                load_form_fields()

            elif method == "Upload Filled Form":

                add_message(
                    "assistant",
                    "Please upload your completed application. "
                    "I'll extract the information using the "
                    "existing document-processing system.",
                )

            elif method == "Upload Government ID":

                add_message(
                    "assistant",
                    "Please upload your government ID. "
                    "I'll extract the information and match it "
                    "against your application fields.",
                )

            st.rerun()

    st.write("")

    if st.button(
        "← Change loan",
        key="change_loan",
        use_container_width=True,
    ):

        st.session_state.loan_type = None

        st.session_state.filling_method = None

        st.session_state.step = 0

        add_message(
            "assistant",
            "Sure. Let's choose another loan type.",
        )

        st.rerun()


# ============================================================
# LOAD FORM
# ============================================================


def load_form_fields():

    loan_name = st.session_state.loan_type

    if not loan_name:
        return

    pdf_name = loan_map[loan_name][0]

    form_path = os.path.join(
        FORMS_FOLDER,
        pdf_name,
    )

    if st.session_state.form_sections is not None:

        update_current_field()

        return

    status = None
    progress = None

    try:

        with st.status(
            "Preparing application form...",
            expanded=True,
        ) as status:

            progress = st.progress(
                0,
                text="Preparing Pages...",
            )

            status.write("Preparing Pages...")

            progress.progress(
                20,
                text="Preparing Pages...",
            )

            status.write("Running OCR Extraction...")

            progress.progress(
                50,
                text="Running OCR Extraction...",
            )

            ocr_text = get_ocr(form_path)

            status.write("Extracting Loan Fields...")

            progress.progress(
                75,
                text="Extracting Loan Fields...",
            )

            extraction = extract_form_fields(ocr_text)

            status.write("Finalizing Results...")

            progress.progress(
                90,
                text="Finalizing Results...",
            )

            sections = parse_form_fields(extraction)

            st.session_state.form_sections = sections

            update_current_field()

            progress.progress(
                100,
                text="Completed ✓",
            )

            status.update(
                label="Completed ✓",
                state="complete",
                expanded=False,
            )

    except Exception as e:

        st.session_state.chat_error = "I couldn't read the application form: " + str(e)


# ============================================================
# MANUAL APPLICATION
# ============================================================


def render_manual_application():

    if st.session_state.form_sections is None:

        load_form_fields()

    if st.session_state.chat_error:

        return

    section, field = update_current_field()

    if field is None:

        st.markdown(
            '<div class="success-box">'
            "<strong>All information collected.</strong><br>"
            "Your application is ready for review."
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Review application →",
            type="primary",
            use_container_width=True,
            key="manual_review",
        ):

            add_message(
                "assistant",
                "Thanks. Your application is ready for review.",
            )

            st.session_state.step = 3

            st.rerun()

        return

    st.markdown(
        f'<div class="system-box">'
        f"<strong>{section}</strong><br><br>"
        f"Please provide your "
        f"<strong>{field}</strong>."
        f"</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT PROCESSING
# ============================================================


def process_user_message(text):

    text = text.strip()

    if not text:
        return

    add_message(
        "user",
        text,
    )

    if (
        st.session_state.step == 2
        and st.session_state.filling_method == "Fill Manually"
    ):

        current_field = st.session_state.current_field

        if current_field:

            set_application_value(
                current_field,
                text,
            )

            section, next_field = update_current_field()

            if next_field:

                add_message(
                    "assistant",
                    f"Thanks. Now, what is your " f"**{next_field}**?",
                )

            else:

                add_message(
                    "assistant",
                    "Thank you. I've collected all the "
                    "application information. "
                    "Let's review it before submission.",
                )

                st.session_state.step = 3

        return

    add_message(
        "assistant",
        "Thanks. Please use the available controls " "to continue your application.",
    )


# ============================================================
# FILLED FORM
# ============================================================


def render_filled_form():

    uploaded_file = st.file_uploader(
        "Upload your completed application",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
        ],
        key="filled_upload",
    )

    if uploaded_file:

        st.success(f"Selected: {uploaded_file.name}")

        if st.button(
            "Extract application information",
            type="primary",
            key="extract_application",
            use_container_width=True,
        ):

            loan_name = st.session_state.loan_type

            form_name = loan_map[loan_name][0]

            form_path = os.path.join(
                FORMS_FOLDER,
                form_name,
            )

            try:

                with st.status(
                    "Processing application...",
                    expanded=True,
                ) as status:

                    progress = st.progress(
                        0,
                        text="Uploading Document...",
                    )

                    # ----------------------------------------
                    # STAGE 1
                    # ----------------------------------------

                    progress.progress(
                        10,
                        text="Uploading Document...",
                    )

                    status.write("Uploading Document...")

                    # ----------------------------------------
                    # STAGE 2
                    # ----------------------------------------

                    progress.progress(
                        20,
                        text="Preparing Pages...",
                    )

                    status.write("Preparing Pages...")

                    # ----------------------------------------
                    # STAGE 3
                    # ----------------------------------------

                    progress.progress(
                        45,
                        text="Running OCR Extraction...",
                    )

                    status.write("Running OCR Extraction...")

                    ocr_text = perform_ocr(uploaded_file)

                    # ----------------------------------------
                    # FORM TEMPLATE
                    # ----------------------------------------

                    progress.progress(
                        60,
                        text="Reading application form...",
                    )

                    status.write("Reading application form...")

                    form_ocr = get_ocr(form_path)

                    form_extraction = extract_form_fields(form_ocr)

                    sections = parse_form_fields(form_extraction)

                    # ----------------------------------------
                    # STAGE 4
                    # ----------------------------------------

                    progress.progress(
                        80,
                        text="Extracting Loan Fields...",
                    )

                    status.write("Extracting Loan Fields...")

                    filled_extraction = extract_filled_form_information(
                        ocr_text,
                        sections,
                    )

                    matched = parse_matched_fields(filled_extraction)

                    # ----------------------------------------
                    # STAGE 5
                    # ----------------------------------------

                    progress.progress(
                        92,
                        text="Finalizing Results...",
                    )

                    status.write("Finalizing Results...")

                    st.session_state.filled_form_sections = sections

                    st.session_state.filled_form_values = matched

                    values = {}

                    for _, fields in sections:

                        for field in fields:

                            values[field] = clean_value(
                                matched.get(
                                    field,
                                    "",
                                )
                            )

                    st.session_state.final_form_values = values

                    st.session_state.document_processed = True

                    # ----------------------------------------
                    # COMPLETE
                    # ----------------------------------------

                    progress.progress(
                        100,
                        text="Completed ✓",
                    )

                    status.update(
                        label="Completed ✓",
                        state="complete",
                        expanded=False,
                    )

                add_message(
                    "user",
                    f"Uploaded {uploaded_file.name}",
                )

                add_message(
                    "assistant",
                    "I've extracted the application information. "
                    "Please review the details below and correct "
                    "anything that needs attention.",
                )

                st.rerun()

            except Exception as e:

                st.session_state.chat_error = (
                    "I couldn't extract the application: " + str(e)
                )

    if st.session_state.filled_form_sections and st.session_state.filled_form_values:

        render_extracted_information(
            st.session_state.filled_form_sections,
            st.session_state.filled_form_values,
            "filled",
        )


# ============================================================
# GOVERNMENT ID
# ============================================================


def render_government_id():

    uploaded_file = st.file_uploader(
        "Upload your government ID",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
        ],
        key="id_upload",
    )

    if uploaded_file:

        st.success(f"Selected: {uploaded_file.name}")

        if st.button(
            "Extract ID information",
            type="primary",
            key="extract_id",
            use_container_width=True,
        ):

            loan_name = st.session_state.loan_type

            form_name = loan_map[loan_name][0]

            form_path = os.path.join(
                FORMS_FOLDER,
                form_name,
            )

            try:

                with st.status(
                    "Processing government ID...",
                    expanded=True,
                ) as status:

                    progress = st.progress(
                        0,
                        text="Uploading Document...",
                    )

                    # ----------------------------------------
                    # STAGE 1
                    # ----------------------------------------

                    progress.progress(
                        10,
                        text="Uploading Document...",
                    )

                    status.write("Uploading Document...")

                    # ----------------------------------------
                    # STAGE 2
                    # ----------------------------------------

                    progress.progress(
                        20,
                        text="Preparing Pages...",
                    )

                    status.write("Preparing Pages...")

                    # ----------------------------------------
                    # STAGE 3
                    # ----------------------------------------

                    progress.progress(
                        45,
                        text="Running OCR Extraction...",
                    )

                    status.write("Running OCR Extraction...")

                    ocr_text = perform_ocr(uploaded_file)

                    # ----------------------------------------
                    # ID EXTRACTION
                    # ----------------------------------------

                    progress.progress(
                        65,
                        text="Extracting Loan Fields...",
                    )

                    status.write("Extracting ID information...")

                    id_extraction = extract_id_information(ocr_text)

                    id_information = parse_id_information(id_extraction)

                    # ----------------------------------------
                    # FORM TEMPLATE
                    # ----------------------------------------

                    progress.progress(
                        75,
                        text="Reading application form...",
                    )

                    status.write("Reading application form...")

                    form_ocr = get_ocr(form_path)

                    form_extraction = extract_form_fields(form_ocr)

                    sections = parse_form_fields(form_extraction)

                    # ----------------------------------------
                    # MATCHING
                    # ----------------------------------------

                    progress.progress(
                        88,
                        text="Matching ID to application...",
                    )

                    status.write("Matching ID to application...")

                    matched_extraction = match_id_to_form(
                        id_information,
                        sections,
                    )

                    matched = parse_matched_fields(matched_extraction)

                    # ----------------------------------------
                    # FINALIZE
                    # ----------------------------------------

                    progress.progress(
                        95,
                        text="Finalizing Results...",
                    )

                    status.write("Finalizing Results...")

                    st.session_state.id_form_sections = sections

                    st.session_state.matched_fields = matched

                    values = {}

                    for _, fields in sections:

                        for field in fields:

                            values[field] = clean_value(
                                matched.get(
                                    field,
                                    "",
                                )
                            )

                    st.session_state.final_form_values = values

                    st.session_state.document_processed = True

                    progress.progress(
                        100,
                        text="Completed ✓",
                    )

                    status.update(
                        label="Completed ✓",
                        state="complete",
                        expanded=False,
                    )

                add_message(
                    "user",
                    f"Uploaded {uploaded_file.name}",
                )

                add_message(
                    "assistant",
                    "I've extracted the information from your ID "
                    "and matched it with the application. "
                    "Please review it below.",
                )

                st.rerun()

            except Exception as e:

                st.session_state.chat_error = (
                    "I couldn't process the government ID: " + str(e)
                )

    if st.session_state.id_form_sections and st.session_state.matched_fields:

        render_extracted_information(
            st.session_state.id_form_sections,
            st.session_state.matched_fields,
            "id",
        )


# ============================================================
# EXTRACTED INFORMATION
# ============================================================


def render_extracted_information(
    sections,
    extracted_values,
    prefix,
):

    st.markdown("### Extracted information")

    st.caption(
        "Review the information below. " "You can edit any value before continuing."
    )

    values = {}

    for section_name, fields in sections:

        if not fields:
            continue

        st.markdown(f"**{section_name}**")

        for field in fields:

            value = clean_value(
                extracted_values.get(
                    field,
                    "",
                )
            )

            values[field] = st.text_input(
                field,
                value=value,
                key=(f"{prefix}_" f"{section_name}_" f"{field}"),
            )

    st.session_state.final_form_values = values

    back, confirm = st.columns(2)

    with back:

        if st.button(
            "← Change method",
            key=f"{prefix}_back",
            use_container_width=True,
        ):

            st.session_state.step = 1

            st.session_state.document_processed = False

            st.rerun()

    with confirm:

        if st.button(
            "Confirm information →",
            type="primary",
            key=f"{prefix}_confirm",
            use_container_width=True,
        ):

            st.session_state.extraction_confirmed = True

            st.session_state.step = 3

            add_message(
                "assistant",
                "Thanks. I've saved the extracted information. "
                "Your application is ready for review.",
            )

            st.rerun()


# ============================================================
# APPLICATION SUMMARY
# ============================================================


def render_summary():

    values = st.session_state.final_form_values

    if not values:

        st.warning("There is no application information to review yet.")

        return

    st.markdown("### Application summary")

    st.markdown(
        '<div class="assistant-bubble">'
        "<strong>Your application is ready for review.</strong>"
        "<br><br>"
        "Please check the information below before submitting."
        "</div>",
        unsafe_allow_html=True,
    )

    sections = get_sections()

    if sections:

        for section_name, fields in sections:

            rows = ""

            for field in fields:

                if field not in values:
                    continue

                value = clean_value(values[field])

                if not value:
                    value = "Not provided"

                rows += (
                    '<div class="info-row">'
                    f'<span class="info-key">'
                    f"{field}"
                    f"</span>"
                    f'<span class="info-value">'
                    f"{value}"
                    f"</span>"
                    "</div>"
                )

            if rows:

                st.markdown(
                    f'<div class="info-card">'
                    f'<div class="info-card-title">'
                    f"{section_name}"
                    f"</div>"
                    f"{rows}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    else:

        rows = ""

        for field, value in values.items():

            value = clean_value(value)

            if not value:
                value = "Not provided"

            rows += (
                '<div class="info-row">'
                f'<span class="info-key">'
                f"{field}"
                f"</span>"
                f'<span class="info-value">'
                f"{value}"
                f"</span>"
                "</div>"
            )

        st.markdown(
            f'<div class="info-card">' f"{rows}" f"</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# REVIEW CONTROLS
# ============================================================


def render_review_controls():

    st.markdown(
        '<div class="system-box">'
        "Everything look correct? You can submit the application, "
        "or go back and make changes."
        "</div>",
        unsafe_allow_html=True,
    )

    change, submit = st.columns(2)

    with change:

        if st.button(
            "✏️ Make changes",
            key="make_changes",
            use_container_width=True,
        ):

            add_message(
                "assistant",
                "No problem. Let's go back and make your changes.",
            )

            st.session_state.step = 2

            if st.session_state.filling_method == "Fill Manually":

                update_current_field()

            st.rerun()

    with submit:

        if st.button(
            "Submit application",
            type="primary",
            key="submit_application",
            use_container_width=True,
        ):

            st.session_state.submitted = True

            st.session_state.step = 4

            add_message(
                "user",
                "Submit application",
            )

            add_message(
                "assistant",
                "🎉 Your application has been submitted successfully.",
            )

            st.rerun()


# ============================================================
# SUBMITTED
# ============================================================


def render_submitted():

    st.markdown(
        """
        <div style="
            background-color:#f0fdf4;
            border:1px solid #bbf7d0;
            border-radius:12px;
            padding:18px 20px;
            margin:10px 0 16px 0;
        ">

            <div style="
                color:#166534 !important;
                font-size:18px;
                font-weight:700;
            ">
                🎉 Application submitted successfully.
            </div>

            <div style="
                color:#166534 !important;
                font-size:13px;
                margin-top:6px;
            ">
                Your loan application has been submitted successfully.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button(
        "Start new application",
        type="primary",
        key="new_application",
        use_container_width=True,
    ):

        reset_application()

        st.rerun()


# ============================================================
# RESET APPLICATION
# ============================================================


def reset_application():

    for key, value in defaults.items():

        if isinstance(value, dict):

            st.session_state[key] = {}

        elif isinstance(value, list):

            st.session_state[key] = []

        else:

            st.session_state[key] = value


# ============================================================
# CURRENT STEP
# ============================================================


def render_current_step():

    step = st.session_state.step

    if step == 0:

        render_loan_selection()

    elif step == 1:

        render_filling_method()

    elif step == 2:

        method = st.session_state.filling_method

        if method == "Fill Manually":

            render_manual_application()

        elif method == "Upload Filled Form":

            render_filled_form()

        elif method == "Upload Government ID":

            render_government_id()

        else:

            st.warning(
                "Please choose how you would like " "to provide your information."
            )

    elif step == 3:

        render_summary()

        render_review_controls()

    elif step == 4:

        render_submitted()


# ============================================================
# FORM DIRECTORY VALIDATION
# ============================================================

if not os.path.exists(FORMS_FOLDER):

    st.error("The forms folder was not found.")

    st.stop()


form_files = [
    file for file in os.listdir(FORMS_FOLDER) if file.lower().endswith(".pdf")
]

form_files.sort()


if not form_files:

    st.error("No PDF forms were found in the forms folder.")

    st.stop()


# ============================================================
# INITIALIZE CHAT
# ============================================================

initialize_conversation()


# ============================================================
# HEADER
# ============================================================

render_header()


# ============================================================
# PROGRESS
# ============================================================

render_progress()


# ============================================================
# CHAT
# ============================================================

st.markdown("#### Conversation")

render_chat()


# ============================================================
# ERROR
# ============================================================

if st.session_state.chat_error:

    st.markdown(
        f"""
        <div class="error-box">
            {st.session_state.chat_error}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Try again",
        key="try_again",
        use_container_width=True,
    ):

        st.session_state.chat_error = None

        st.rerun()


# ============================================================
# CURRENT INTERACTION
# ============================================================

render_current_step()


# ============================================================
# CHAT INPUT
# ============================================================

if (
    st.session_state.step == 2
    and st.session_state.filling_method == "Fill Manually"
    and st.session_state.current_field
):

    user_text = st.chat_input(f"Enter your " f"{st.session_state.current_field}...")

    if user_text:

        process_user_message(user_text)

        st.rerun()
