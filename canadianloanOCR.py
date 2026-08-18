import os
import re

import streamlit as st
import pytesseract
import pandas as pd

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pdf2image import convert_from_bytes

load_dotenv()


# display extracted section content


def display_section_content(content):

    i = 0

    while i < len(content):

        line = content[i].strip()

        if not line:
            i += 1
            continue

        # detect table

        if line.startswith("TABLE "):

            st.markdown(f"#### {line}")

            i += 1

            columns = []

            rows = []

            # read columns

            if i < len(content) and content[i].startswith("Columns:"):

                column_text = content[i][len("Columns:") :].strip()

                if column_text:

                    columns = [column.strip() for column in column_text.split("|")]

                i += 1

            # read rows

            while i < len(content):

                current_line = content[i].strip()

                if current_line.startswith("TABLE "):
                    break

                if re.match(r"SECTION\s+\d+\s*:", current_line):
                    break

                if current_line.startswith("UNSECTIONED"):
                    break

                if current_line.startswith("Field:"):
                    break

                if current_line.startswith("Row "):

                    i += 1

                    if i < len(content):

                        row_line = content[i].strip()

                        row_values = [value.strip() for value in row_line.split("|")]

                        rows.append(row_values)

                        i += 1

                    continue

                i += 1

            # display table

            if columns and rows:

                normalized_rows = []

                for row in rows:

                    if len(row) < len(columns):

                        row = row + ["Empty/Not Filled"] * (len(columns) - len(row))

                    elif len(row) > len(columns):

                        row = row[: len(columns)]

                    normalized_rows.append(row)

                try:

                    table_df = pd.DataFrame(normalized_rows, columns=columns)

                    st.dataframe(table_df, use_container_width=True, hide_index=True)

                except Exception:

                    st.write("Table structure could not be reconstructed.")

            else:

                st.write("Table structure could not be reconstructed.")

            st.divider()

            continue

        # display fields

        if line.startswith("Field:"):

            st.markdown(f"**{line}**")

        elif line.startswith("Value:"):

            st.write(line)

        else:

            st.write(line)

        i += 1


# parse model output into sections


def parse_extraction(extraction):

    sections = []

    current_section = None
    current_content = []

    unsectioned_content = []

    lines = extraction.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # remove unnecessary headings

        if line.startswith("Number of Sections:"):

            continue

        # detect numbered section

        section_match = re.match(r"^SECTION\s+(\d+)\s*:\s*(.+)$", line, re.IGNORECASE)

        if section_match:

            if current_section:

                sections.append((current_section, current_content))

            current_section = line
            current_content = []

            continue

        # fallback for section without number

        if line.upper().startswith("SECTION:"):

            if current_section:

                sections.append((current_section, current_content))

            current_section = line
            current_content = []

            continue

        # unsectioned information

        if line.upper().startswith("UNSECTIONED INFORMATION"):

            if current_section:

                sections.append((current_section, current_content))

            current_section = "SECTION 1: UNSECTIONED INFORMATION"

            current_content = []

            continue

        # store content

        if current_section:

            current_content.append(line)

        else:

            unsectioned_content.append(line)

    # add final section

    if current_section:

        sections.append((current_section, current_content))

    # add information before first section

    if unsectioned_content:

        sections.insert(0, ("SECTION 1: UNSECTIONED INFORMATION", unsectioned_content))

    # renumber sections sequentially

    final_sections = []

    for index, (section_name, content) in enumerate(sections, start=1):

        if section_name.upper().startswith("SECTION"):

            if ":" in section_name:

                section_title = section_name.split(":", 1)[1].strip()

            else:

                section_title = section_name

        else:

            section_title = section_name

        new_section_name = f"SECTION {index}: {section_title}"

        final_sections.append((new_section_name, content))

    return final_sections


# clean OCR text


def clean_ocr_text(text):

    lines = text.splitlines()

    cleaned_lines = []

    previous_line = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # remove repeated OCR lines

        if line == previous_line:
            continue

        cleaned_lines.append(line)

        previous_line = line

    return "\n".join(cleaned_lines)


# streamlit ui

st.title("Canadian Loan Form Information Extractor")

st.write(
    "Upload a Canadian loan form PDF. The system "
    "will identify sections, selected fields, "
    "handwritten values, checkboxes, and tables."
)


# file upload

uploaded_file = st.file_uploader("Upload a Canadian loan form PDF", type=["pdf"])


if uploaded_file is not None:

    # read pdf

    try:

        pdf_bytes = uploaded_file.read()

        pdf_pages = convert_from_bytes(pdf_bytes, dpi=180)

    except Exception as e:

        st.error(f"Error while reading PDF: {e}")

        st.stop()

    st.write(f"**Selected file:** {uploaded_file.name}")

    st.write(f"**Number of pages:** {len(pdf_pages)}")

    # extract information

    if st.button("Extract Information"):

        # get hugging face token

        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:

            st.error("HF_TOKEN was not found in the .env file.")

            st.info("Add HF_TOKEN=hf_your_token_here " "to your .env file.")

            st.stop()

        # connect to hugging face

        with st.spinner("connecting to the hugging face ai model..."):

            try:

                llm = HuggingFaceEndpoint(
                    repo_id=("Qwen/Qwen3-4B-Instruct-2507"),
                    task="text-generation",
                    provider="auto",
                    huggingfacehub_api_token=hf_token,
                    max_new_tokens=3500,
                    temperature=0.0,
                )

                chat = ChatHuggingFace(llm=llm)

            except Exception as e:

                st.error(f"Error while connecting to Hugging Face: {e}")

                st.stop()

        # store OCR

        all_page_ocr = []

        # progress

        progress_bar = st.progress(0)

        status_message = st.empty()

        # process each pdf page

        for page_number, image in enumerate(pdf_pages, start=1):

            status_message.info(
                f"processing page {page_number} " f"of {len(pdf_pages)}..."
            )

            # image preprocessing

            try:

                status_message.info(
                    f"improving image quality " f"for page {page_number}..."
                )

                gray_image = ImageOps.grayscale(image)

                width, height = gray_image.size

                scale = 2

                gray_image = gray_image.resize(
                    (width * scale, height * scale), Image.Resampling.LANCZOS
                )

                contrast_image = ImageEnhance.Contrast(gray_image).enhance(1.8)

                sharpened_image = contrast_image.filter(ImageFilter.SHARPEN)

                sharpened_image = sharpened_image.filter(
                    ImageFilter.UnsharpMask(radius=2, percent=130, threshold=3)
                )

            except Exception as e:

                st.warning(f"Could not improve page " f"{page_number}: {e}")

                continue

            # OCR

            status_message.info(
                f"reading text from page " f"{page_number} of " f"{len(pdf_pages)}..."
            )

            try:

                # layout preserving OCR

                layout_text = pytesseract.image_to_string(
                    sharpened_image,
                    config=("--psm 6 " "-c preserve_interword_spaces=1"),
                )

                # normal OCR

                standard_text = pytesseract.image_to_string(
                    sharpened_image, config="--psm 6"
                )

                # sparse text OCR

                sparse_text = pytesseract.image_to_string(
                    sharpened_image, config="--psm 11"
                )

            except Exception as e:

                st.warning(f"Could not read page " f"{page_number}: {e}")

                continue

            # clean OCR

            layout_text = clean_ocr_text(layout_text)

            standard_text = clean_ocr_text(standard_text)

            sparse_text = clean_ocr_text(sparse_text)

            page_text = (
                f"\n\n===== PAGE {page_number} =====\n\n"
                f"LAYOUT OCR:\n"
                f"{layout_text}\n\n"
                f"STANDARD OCR:\n"
                f"{standard_text}\n\n"
                f"SPARSE OCR:\n"
                f"{sparse_text}"
            )

            all_page_ocr.append(page_text)

            progress_bar.progress(page_number / len(pdf_pages))

        # check OCR

        combined_ocr = "\n".join(all_page_ocr)

        if not combined_ocr.strip():

            st.error("No text could be extracted from the PDF.")

            st.stop()

        # avoid sending excessive duplicate OCR

        max_ocr_characters = 70000

        if len(combined_ocr) > max_ocr_characters:

            st.warning(
                "The PDF contains a large amount of OCR "
                "text. The OCR has been limited to keep "
                "the Hugging Face request within the "
                "model context limit."
            )

            combined_ocr = combined_ocr[:max_ocr_characters]

        # show OCR

        with st.expander("view extracted ocr text"):

            st.text(combined_ocr)

        # LLM extraction

        status_message.info(
            "extracting information from the complete " "document using hugging face..."
        )

        extraction_prompt = f"""
You are an expert Canadian loan-document
information extraction assistant.

Analyze the complete OCR text of a Canadian
loan application or financing PDF.

The OCR comes from multiple pages.

Your job is to reconstruct the structure of the
actual document as accurately as possible.

Do not invent information.

Do not guess information.

Do not create information that is not present.

==================================================
SECTION IDENTIFICATION
==================================================

Identify EVERY actual section present in the
document.

For every section:

1. Assign a sequential section number.
2. Preserve the actual section name.
3. Keep the original document order.
4. Put fields under the correct section.

Use this exact format:

SECTION 1: Applicant Information

SECTION 2: Employment Information

SECTION 3: Loan Information

SECTION 4: Banking Information

Do not invent section names.

Do not create artificial sections.

If information appears before the first named
section, use:

SECTION 1: UNSECTIONED INFORMATION

==================================================
IMPORTANT FIELD RULE
==================================================

Only return fields that are actually relevant
and present in the document.

Do NOT output generic loan fields simply because
they are common on loan applications.

For example, if the document does not contain
an employer field, do not create:

Field: Employer
Value: Empty/Not Filled

Do not create empty placeholder fields.

Do not create fields from a standard template.

Only extract fields that actually appear on the
page/document.

==================================================
CHECKBOX RULE
==================================================

Checkboxes are extremely important.

If a group contains options such as:

[ ] Personal Loan
[x] Vehicle Loan
[ ] Mortgage

ONLY return the selected option:

Field: Loan Type
Value: Vehicle Loan

Do NOT return:

Personal Loan - Not Selected
Mortgage - Not Selected

Do NOT output unselected options.

If none of the options are selected, omit the
checkbox group unless the document explicitly
indicates that no option was selected.

If the checkbox state cannot be determined,
use:

Cannot Determine

Do not guess.

==================================================
FIELD VALUE RULE
==================================================

Extract a field only when it has an actual value
or when the field is clearly selected.

Examples:

Field: First Name
Value: John

Field: Loan Amount
Value: $10,000

Field: Employment Status
Value: Full Time

Do not create empty fields.

Do not create placeholder values.

Do not infer missing information.

==================================================
HANDWRITTEN VALUES
==================================================

The document may contain handwriting.

Handwritten information is a valid value when it
appears inside or beside a corresponding field.

Extract the handwritten value as accurately as
possible.

Do not replace handwriting with assumptions.

If handwriting cannot be read:

Cannot Determine

==================================================
CANADIAN INFORMATION
==================================================

Extract Canadian information when actually
present, such as:

Name
Date of Birth
Address
City
Province
Postal Code
Phone
Email
SIN
Employment
Employer
Income
Expenses
Assets
Liabilities
Banking
Loan Information
Vehicle Information
References
Consent
Authorization
Signature

These are examples only.

DO NOT create these fields unless they actually
appear in the document.

==================================================
LOAN TYPE
==================================================

Extract the loan type only if explicitly present.

Examples:

Personal Loan
Vehicle Loan
Car Loan
Mortgage
Home Loan
Business Loan
Education Loan
Debt Consolidation
Line of Credit
Consumer Financing
Point-of-Sale Financing
Secured Loan
Unsecured Loan

Do not infer the loan type.

==================================================
COMPANY / BRAND
==================================================

If a company or brand is explicitly visible,
extract it.

Examples:

goeasy
easyfinancial
easyhome
LendCare

Do not assume the company.

==================================================
TABLE EXTRACTION
==================================================

Tables must be preserved as tables.

Do NOT flatten tables into normal fields.

Identify:

- table title
- columns
- rows
- row labels
- cell values
- totals
- subtotals
- selected checkboxes

Maintain the original row and column relationships.

==================================================
TABLE FORMAT
==================================================

Use exactly:

TABLE 1: Income Information

Columns:
Description | Amount | Frequency

Row 1:
Salary | $5,000 | Monthly

Row 2:
Rent | $1,500 | Monthly

Row 3:
Other Income | $500 | Monthly

If a cell is genuinely empty:

Empty/Not Filled

If the cell cannot be read:

Cannot Determine

Do not move values between columns.

Do not move values between rows.

Do not invent rows.

Do not invent columns.

==================================================
MULTI-PAGE TABLES
==================================================

A table may continue onto another page.

Keep continuation rows associated with the
correct table when the relationship is clear.

Do not invent missing headers.

==================================================
OUTPUT FORMAT
==================================================

Do NOT return JSON.

Do NOT use Markdown tables.

Do NOT provide explanations.

Return ONLY the extracted information.

Use this structure:

SECTION 1: [Actual Section Name]

Field: [Actual Field Name]
Value: [Actual Value]

Field: [Actual Field Name]
Value: [Actual Value]

TABLE 1: [Actual Table Name]

Columns:
Column 1 | Column 2 | Column 3

Row 1:
Value 1 | Value 2 | Value 3

Row 2:
Value 1 | Value 2 | Value 3


SECTION 2: [Actual Section Name]

Field: [Actual Field Name]
Value: [Actual Value]

Continue for EVERY actual section.

==================================================
CRITICAL FINAL RULES
==================================================

1. Only extract information actually present.

2. Do not create generic fields.

3. Do not create empty fields.

4. Do not output unselected checkbox options.

5. Only output selected checkbox options.

6. Preserve handwritten values.

7. Preserve tables.

8. Preserve row and column relationships.

9. Preserve the original section names.

10. Number every section sequentially.

11. Do not guess.

12. Do not omit actual filled information.

==================================================
OCR DOCUMENT
==================================================

{combined_ocr}
"""

        # invoke model once

        try:

            response = chat.invoke(extraction_prompt)

            extracted_information = response.content

        except Exception as e:

            st.error(f"Error while extracting information: {e}")

            if "402" in str(e):

                st.warning(
                    "Your Hugging Face inference " "credits appear to be exhausted."
                )

            elif "403" in str(e):

                st.warning(
                    "Your Hugging Face token does not "
                    "have permission to use the selected "
                    "Inference Provider."
                )

            st.stop()

        # extraction finished

        status_message.success("information extraction completed successfully.")

        progress_bar.progress(1.0)

        # parse sections

        sections = parse_extraction(extracted_information)

        if not sections:

            st.warning("No sections could be identified.")

            with st.expander("view raw model response"):

                st.text(extracted_information)

            st.stop()

        # display result

        st.subheader("Extracted Loan Information")

        st.success(f"identified {len(sections)} section(s)")

        # display sections

        for section_name, content in sections:

            if not content:
                continue

            # get section title

            if ":" in section_name:

                section_title = section_name.split(":", 1)[1].strip()

            else:

                section_title = section_name

            # section heading

            st.markdown(f"## {section_name}")

            st.divider()

            # section content

            display_section_content(content)

        # complete extraction

        with st.expander("view complete extraction"):

            st.text(extracted_information)
