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

                if re.match(r"SECTION\s+\d+\s*:", current_line, re.IGNORECASE):
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

        elif line.startswith("Status:"):

            status_value = line[len("Status:") :].strip()

            if status_value == "Selected":

                st.success(line)

            elif status_value == "Not Selected":

                st.info(line)

            elif status_value == "Cannot Determine":

                st.warning(line)

            else:

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

        # remove loan type from section parsing

        if line.upper().startswith("LOAN TYPE:"):

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


# extract loan type from model output


def extract_loan_type(extraction):

    for line in extraction.splitlines():

        line = line.strip()

        if line.upper().startswith("LOAN TYPE:"):

            loan_type = line.split(":", 1)[1].strip()

            if loan_type:

                return loan_type

    return "Cannot Determine"


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

st.title("Loan Information Extractor")

st.write(
    "Upload a loan form PDF. The system "
    "will identify the loan type, sections, "
    "fields, filled and empty fields, "
    "checkbox selections, handwritten values, "
    "and tables."
)


# file upload

uploaded_file = st.file_uploader("Upload a loan form PDF", type=["pdf"])


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

                st.error(f"Error while connecting to " f"Hugging Face: {e}")

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
                f"\n\n===== PAGE "
                f"{page_number} =====\n\n"
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

            st.error("No text could be extracted " "from the PDF.")

            st.stop()

        # avoid sending excessive duplicate OCR

        max_ocr_characters = 70000

        if len(combined_ocr) > max_ocr_characters:

            st.warning(
                "The PDF contains a large amount "
                "of OCR text. The OCR has been "
                "limited to keep the Hugging Face "
                "request within the model context "
                "limit."
            )

            combined_ocr = combined_ocr[:max_ocr_characters]

        # show OCR

        with st.expander("view extracted ocr text"):

            st.text(combined_ocr)

        # LLM extraction

        status_message.info(
            "extracting information from the "
            "complete document using "
            "hugging face..."
        )

        extraction_prompt = f"""
You are an expert loan-document information
extraction assistant.

Analyze the complete OCR text of a loan
application, financing form, or loan-related PDF.

The document may be:

1. A common/general loan application form
   containing options for multiple loan types.

OR

2. A specialized form designed for one
   particular loan type.

Your job is to accurately identify the loan
type and extract ALL information that is
actually present on the document.

The OCR comes from multiple pages.

Do not invent information.

Do not guess information.

Do not create information that is not present.

==================================================
LOAN TYPE CLASSIFICATION
==================================================

First identify what type of loan the document
represents.

Determine the loan type using ONLY information
actually present in the document.

Look at:

- document title
- form title
- headings
- loan-related labels
- selected checkboxes
- selected options
- descriptions
- product names
- vehicle-related information
- mortgage/property-related information
- business-related information
- education-related information
- debt-consolidation information
- explicitly stated loan type

Possible examples include:

Personal Loan
Vehicle Loan
Auto Loan
Car Loan
Mortgage
Home Loan
Business Loan
Education Loan
Student Loan
Debt Consolidation Loan
Line of Credit
Consumer Financing
Point-of-Sale Financing
Secured Loan
Unsecured Loan
Other

These are examples only.

Do NOT assume that a loan belongs to one of
these categories just because it is common.

If the document explicitly states the loan type,
use that information.

If the document contains multiple loan-type
checkboxes or options, determine which option
is selected.

If exactly one loan type is selected, classify
the document as that loan type.

If multiple loan types are selected, report all
selected loan types.

If the document is clearly a specialized form
for one loan type, classify it according to the
actual document.

For example, if the document clearly contains
vehicle-financing information such as:

Vehicle Make
Vehicle Model
Vehicle Year
VIN
Purchase Price
Vehicle Financing

and the document clearly represents vehicle
financing, the loan type may be classified as:

LOAN TYPE: Vehicle Loan

However, do not guess when the evidence is
insufficient.

If the loan type cannot be determined from the
document, output:

LOAN TYPE: Cannot Determine

Use exactly:

LOAN TYPE: [loan type]


==================================================
IMPORTANT: INFORMATION BEFORE THE FIRST SECTION
==================================================

The document may contain important information
BEFORE the first formal section heading.

This information MUST NOT be ignored.

For example, the top of the document may contain:

Loan Requested: $25,000

Loan Type: Vehicle Loan

Requested Term: 60 Months

Application Date: 2026-08-18

Application Number: 12345

Applicant Information

First Name: John

In this example:

Loan Requested
Loan Type
Requested Term
Application Date
Application Number

are actual document fields appearing before the
first formal section.

These fields MUST be extracted.

If genuine fields appear before the first formal
section, place them under:

SECTION 1: UNSECTIONED INFORMATION

For example:

SECTION 1: UNSECTIONED INFORMATION

Field: Loan Requested
Value: $25,000

Field: Loan Type
Value: Vehicle Loan

Field: Requested Term
Value: 60 Months

Field: Application Date
Value: 2026-08-18

Field: Application Number
Value: 12345


Then continue with the actual sections:

SECTION 2: Applicant Information

Field: First Name
Value: John


==================================================
DO NOT IGNORE TOP-OF-DOCUMENT INFORMATION
==================================================

Information appearing before the first formal
section is still part of the document.

Do NOT assume that information is irrelevant
because it appears at the top of the page.

Pay special attention to fields such as:

Loan Requested
Requested Amount
Loan Amount
Loan Type
Requested Term
Loan Term
Application Date
Date Requested
Application Number
Application ID
Reference Number
Dealer
Branch
Product
Financing Type
Account Number

These are examples only.

DO NOT create these fields unless they actually
appear on the document.

Only extract fields that are physically present
in the uploaded document.


==================================================
DO NOT TREAT HEADINGS AS FIELDS
==================================================

Do not convert document titles, company names,
logos, decorative text, or section headings into
fields unless they clearly represent an actual
field or piece of information.

For example:

ABC Financial Services
LOAN APPLICATION
APPLICANT INFORMATION

should NOT automatically become:

Field: Company
Value: ABC Financial Services

Field: Document Type
Value: Loan Application

Only extract them if the document clearly
presents them as actual information or a field.

However, if the document explicitly provides
information such as:

Lender: ABC Financial Services

then extract:

Field: Lender
Value: ABC Financial Services


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

Do not create sections simply because a field
belongs to a common category.

Use the actual section headings present in
the document whenever possible.


==================================================
SECTION ORDER
==================================================

Preserve the order of information in the
original document.

If information appears before the first formal
section, it must appear first.

For example:

LOAN TYPE: Vehicle Loan

SECTION 1: UNSECTIONED INFORMATION

Field: Loan Requested
Value: $25,000

Field: Requested Term
Value: 60 Months

SECTION 2: Applicant Information

Field: First Name
Value: John

SECTION 3: Vehicle Information

Field: Vehicle Make
Value: Toyota


==================================================
IMPORTANT FIELD RULE
==================================================

Only return fields that are ACTUALLY PRESENT
ON THE DOCUMENT.

This is extremely important.

The system must identify fields that physically
appear on the uploaded form.

Do NOT use a predefined list of standard
loan fields.

Do NOT output generic loan fields simply because
they are common on loan applications.

For example, if the document does not contain
an employer field, DO NOT create:

Field: Employer
Value: Empty/Not Filled

The field must actually appear on the document.

If a field physically appears on the form but
has no value entered, report it as:

Field: Employer
Value: Empty/Not Filled


==================================================
EMPTY FIELD RULE
==================================================

You MUST identify empty fields that are actually
printed on the document.

For every actual field physically present:

1. If the field has a readable value:

Field: [Actual Field Name]
Value: [Actual Value]

2. If the field is present but empty:

Field: [Actual Field Name]
Value: Empty/Not Filled

3. If the field contains something but the value
   cannot be read:

Field: [Actual Field Name]
Value: Cannot Determine

Example:

First Name: John
Last Name: __________
Date of Birth: __________
Phone: 555-1234

Return:

Field: First Name
Value: John

Field: Last Name
Value: Empty/Not Filled

Field: Date of Birth
Value: Empty/Not Filled

Field: Phone
Value: 555-1234


==================================================
DO NOT CREATE MISSING FIELDS
==================================================

If a field does NOT physically exist anywhere
on the uploaded document, do NOT output it.

For example, if there is no SIN field on the
document, do NOT output:

Field: SIN
Value: Empty/Not Filled

Only fields actually present on the document
should be extracted.

This rule applies to every type of field.


==================================================
CHECKBOX RULE
==================================================

Checkboxes are extremely important.

You must identify checkbox groups that are
actually present on the document.

For EVERY checkbox option that is physically
present, determine its state.

Possible states are:

Selected
Not Selected
Cannot Determine


If selected:

Field: Loan Type
Value: Vehicle Loan
Status: Selected


If not selected:

Field: Personal Loan
Value: Not Selected
Status: Not Selected


If the checkbox is present but its state cannot
be determined:

Field: Business Loan
Value: Cannot Determine
Status: Cannot Determine


Do NOT omit an unselected checkbox if the
checkbox is physically present on the form.

Do NOT create checkbox options that are not
physically present on the form.

Do NOT assume a checkbox is selected.

Use only available OCR and document evidence.


==================================================
CHECKBOX GROUP EXAMPLE
==================================================

If the document contains:

[ ] Personal Loan
[x] Vehicle Loan
[ ] Mortgage

Return:

Field: Personal Loan
Value: Not Selected
Status: Not Selected

Field: Vehicle Loan
Value: Vehicle Loan
Status: Selected

Field: Mortgage
Value: Not Selected
Status: Not Selected


==================================================
CHECKBOX AND LOAN TYPE
==================================================

If a checkbox group represents loan types,
use the selected checkbox to help classify
the loan.

For example:

[ ] Personal Loan
[x] Vehicle Loan
[ ] Mortgage

Then:

LOAN TYPE: Vehicle Loan

and also report the checkbox states:

Field: Personal Loan
Value: Not Selected
Status: Not Selected

Field: Vehicle Loan
Value: Vehicle Loan
Status: Selected

Field: Mortgage
Value: Not Selected
Status: Not Selected


If none of the loan-type checkboxes are
selected, do NOT guess the loan type.

Use other explicit evidence from the document.

If there is still insufficient evidence:

LOAN TYPE: Cannot Determine


==================================================
HANDWRITTEN VALUES
==================================================

The document may contain handwriting.

Handwritten information is a valid value when
it appears inside or beside a corresponding
field.

Extract the handwritten value as accurately
as possible.

Do not replace handwriting with assumptions.

If handwriting exists but cannot be read:

Field: [Actual Field Name]
Value: Cannot Determine


==================================================
SIGNATURES
==================================================

If a signature field is physically present,
identify whether it appears to contain a
signature when possible.

For example:

Field: Applicant Signature
Value: Signed

or:

Field: Applicant Signature
Value: Empty/Not Filled

If it is impossible to determine:

Field: Applicant Signature
Value: Cannot Determine

Do not invent a person's name from a signature.


==================================================
DATES
==================================================

Extract dates exactly as they appear when
possible.

Do not change a date into another date.

For example:

Field: Application Date
Value: 2026-08-18

If the date field is present but empty:

Field: Application Date
Value: Empty/Not Filled


==================================================
NUMBERS AND CURRENCY
==================================================

Preserve monetary values and numbers accurately.

For example:

Field: Loan Requested
Value: $25,000

Field: Monthly Payment
Value: $450

Field: Loan Term
Value: 60 Months

Do not calculate missing values.

Do not infer amounts.


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

These are examples only.

Do not assume the company.

Only extract a company or brand if it actually
appears on the document.


==================================================
CANADIAN INFORMATION
==================================================

If information such as the following is actually
present on the document, extract it:

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
appear on the document.


==================================================
LOAN INFORMATION
==================================================

Extract all actual loan-related fields present
on the document.

Examples include:

Loan Requested
Loan Amount
Requested Amount
Loan Type
Loan Term
Interest Rate
Monthly Payment
Payment Frequency
Purpose of Loan
Financing Type
Down Payment
Purchase Price
Amount Financed

These are examples only.

Do NOT create these fields unless they actually
appear on the document.


==================================================
VEHICLE INFORMATION
==================================================

If vehicle-related fields are present, extract
them.

Examples:

Vehicle Make
Vehicle Model
Vehicle Year
VIN
Vehicle Price
Purchase Price
Down Payment
Mileage
Dealer

These are examples only.

Do NOT create these fields unless they actually
appear on the document.


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

Maintain the original row and column
relationships.


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
EMPTY TABLE CELLS
==================================================

If a table cell is physically present but
contains no value, report:

Empty/Not Filled

Do not remove the row because one cell
is empty.

For example:

Row 2:
Rent | Empty/Not Filled | Monthly

Preserve the relationship between the row
and its columns.


==================================================
MULTI-PAGE TABLES
==================================================

A table may continue onto another page.

Keep continuation rows associated with the
correct table when the relationship is clear.

Do not invent missing headers.

Do not create a new table simply because
the table continues on another page.


==================================================
MULTI-PAGE DOCUMENT
==================================================

The OCR contains information from multiple
pages.

Analyze the COMPLETE OCR document before
determining:

- loan type
- sections
- fields
- empty fields
- checkbox states
- tables

Do not classify the document based only on
the first page.

Information on later pages may determine
the actual loan type.


==================================================
DUPLICATE OCR INFORMATION
==================================================

The OCR may contain multiple representations
of the same page:

LAYOUT OCR
STANDARD OCR
SPARSE OCR

These may contain duplicate information.

Do not extract the same field multiple times
just because it appears in multiple OCR versions.

Combine the evidence and return each actual
field only once.


==================================================
OCR ERRORS
==================================================

OCR may contain spelling or character errors.

Use surrounding document context to understand
obvious OCR errors.

However:

DO NOT invent values.

DO NOT guess unreadable handwriting.

DO NOT guess missing numbers.

If the value cannot be reliably determined:

Cannot Determine


==================================================
SECTION CONTENT
==================================================

Place each field under the section where it
actually appears.

Do not move fields between sections simply
because they are logically related.

For example, if "Loan Requested" appears before
the first section, keep it under:

SECTION 1: UNSECTIONED INFORMATION

Do not move it into a later "Loan Information"
section unless it actually appears there.


==================================================
OUTPUT FORMAT
==================================================

Do NOT return JSON.

Do NOT use Markdown tables.

Do NOT provide explanations.

Return ONLY the extracted information.

Start with:

LOAN TYPE: [Loan Type]

Then return the information in document order.

If fields exist before the first formal section,
return:

SECTION 1: UNSECTIONED INFORMATION

Field: Loan Requested
Value: $25,000

Field: Requested Term
Value: 60 Months

Then continue with the actual sections.

Example:

LOAN TYPE: Vehicle Loan

SECTION 1: UNSECTIONED INFORMATION

Field: Loan Requested
Value: $25,000

Field: Requested Term
Value: 60 Months

Field: Application Date
Value: 2026-08-18


SECTION 2: Applicant Information

Field: First Name
Value: John

Field: Last Name
Value: Empty/Not Filled

Field: Date of Birth
Value: 1995-05-10


SECTION 3: Loan Information

Field: Loan Purpose
Value: Vehicle Purchase

Field: Personal Loan
Value: Not Selected
Status: Not Selected

Field: Vehicle Loan
Value: Vehicle Loan
Status: Selected

Field: Mortgage
Value: Not Selected
Status: Not Selected


TABLE 1: Income Information

Columns:
Description | Amount | Frequency

Row 1:
Salary | $5,000 | Monthly

Row 2:
Other Income | Empty/Not Filled | Monthly


Continue for EVERY actual section, field,
checkbox, and table present in the document.


==================================================
CRITICAL FINAL RULES
==================================================

1. Analyze the complete document.

2. Identify the loan type using actual evidence
   from the document.

3. Do not guess the loan type.

4. If the loan type cannot be determined:

   LOAN TYPE: Cannot Determine

5. Extract actual fields that appear before the
   first formal section.

6. Do NOT ignore information at the top of the
   document.

7. Fields before the first formal section must
   be placed under:

   SECTION 1: UNSECTIONED INFORMATION

8. Only extract fields physically present on
   the document.

9. Do not use a predefined list of fields.

10. Do not create generic fields.

11. If an actual field is empty:

    Empty/Not Filled

12. If an actual field cannot be read:

    Cannot Determine

13. Do not create fields that are not present.

14. Report checkbox options that are physically
    present.

15. Report selected checkbox options.

16. Report unselected checkbox options.

17. Report checkbox state as:

    Selected
    Not Selected
    Cannot Determine

18. Do not create checkbox options that are
    not present.

19. Use selected loan-type checkboxes to help
    classify the loan.

20. Preserve handwritten values.

21. Do not guess handwritten values.

22. Preserve tables.

23. Preserve empty table cells.

24. Preserve row and column relationships.

25. Preserve actual section names.

26. Preserve document order.

27. Do not move fields into sections where
    they did not appear.

28. Do not duplicate fields because the same
    information appears in multiple OCR versions.

29. Do not omit actual filled information.

30. Do not omit actual empty fields.

31. Do not return JSON.

32. Do not use Markdown tables.

33. Do not provide explanations.

34. Return ONLY the requested extraction.


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

            st.error(f"Error while extracting " f"information: {e}")

            if "402" in str(e):

                st.warning(
                    "Your Hugging Face inference " "credits appear to be exhausted."
                )

            elif "403" in str(e):

                st.warning(
                    "Your Hugging Face token does "
                    "not have permission to use the "
                    "selected Inference Provider."
                )

            st.stop()

        # extraction finished

        status_message.success("information extraction completed successfully.")

        progress_bar.progress(1.0)

        # extract loan type

        loan_type = extract_loan_type(extracted_information)

        # display loan type

        st.subheader("Loan Classification")

        if loan_type == "Cannot Determine":

            st.warning("Loan Type: Cannot Determine")

        else:

            st.success(f"Loan Type: {loan_type}")

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
