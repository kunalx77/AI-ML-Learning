import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# load environment variables

load_dotenv()


# create ai model


def create_model():

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:

        raise ValueError("HF_TOKEN was not found in the .env file.")

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        task="text-generation",
        provider="nscale",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=2500,
        temperature=0.0,
    )

    return ChatHuggingFace(llm=llm)


# extract form fields


def extract_form_fields(ocr_text):

    chat = create_model()

    prompt = f"""
You are an expert document form structure extraction
assistant.

You are given OCR text from a loan application form.

Your job is to identify the COMPLETE structure of the
form.

The form may contain:

- document title
- loan information
- fields before the first section
- section headings
- fields inside sections
- checkbox options
- tables
- signature fields
- additional information

==================================================
IMPORTANT: EXTRACT THE COMPLETE FORM
==================================================

Extract EVERY actual field that physically appears
on the document.

Do not create fields that are not present.

Do not use a predefined list of common loan fields.

Do not assume fields exist just because they are
common in loan applications.

Only extract fields that can actually be identified
from the OCR/document.

==================================================
FIELDS BEFORE THE FIRST SECTION
==================================================

This is extremely important.

The form may contain actual fields BEFORE the first
formal section heading.

These fields MUST be extracted.

For example, the document may look like:

Loan Application

Loan Requested

Loan Amount Requested ($)
Purpose of Loan
Preferred Repayment Term

New Purchase
Resale Property
Construction
Home Improvement

Applicant Information

Full Legal Name
Date of Birth
Street Address

In this example:

Loan Amount Requested ($)
Purpose of Loan
Preferred Repayment Term
New Purchase
Resale Property
Construction
Home Improvement

must NOT be ignored just because they appear before
"Applicant Information".

Return them BEFORE the first SECTION.

Use:

FIELD: Loan Amount Requested ($)
FIELD: Purpose of Loan
FIELD: Preferred Repayment Term

Then:

SECTION: Applicant Information

FIELD: Full Legal Name
FIELD: Date of Birth
FIELD: Street Address

==================================================
UNSECTIONED INFORMATION
==================================================

If actual fields appear before the first formal
section heading, keep them as unsectioned fields.

Do NOT create an artificial section name in the
extraction.

Simply return the fields first:

FIELD: Loan Amount Requested ($)
FIELD: Purpose of Loan
FIELD: Preferred Repayment Term

SECTION: Applicant Information

FIELD: Full Legal Name

The application will handle the unsectioned fields
separately.

==================================================
SECTION IDENTIFICATION
==================================================

Identify every actual section heading present
in the document.

Preserve the original section name as closely
as possible.

For example:

SECTION: Applicant Information

FIELD: Full Legal Name
FIELD: Date of Birth

SECTION: Employment & Income

FIELD: Employer / Organization Name
FIELD: Job Title / Position

Do not invent section names.

Do not create sections simply because fields
belong logically together.

Only use actual headings present on the document.

==================================================
FIELD IDENTIFICATION
==================================================

A field is something on the form that expects
information, a selection, a response, or a signature.

Examples:

Full Legal Name
Date of Birth
Phone Number
Email Address
Loan Amount Requested ($)
Purpose of Loan
Preferred Repayment Term
Employer
Job Title
Property Address

These are examples only.

Do NOT automatically create these fields.

Only return fields that actually appear on the
document.

==================================================
CHECKBOXES AND OPTIONS
==================================================

Checkbox options are also actual form fields when
they appear on the document.

For example:

[ ] New Purchase
[ ] Resale Property
[ ] Construction
[ ] Home Improvement

Return:

FIELD: New Purchase
FIELD: Resale Property
FIELD: Construction
FIELD: Home Improvement

Do not omit checkbox options.

Do not create checkbox options that are not visible
on the document.

==================================================
RADIO BUTTONS / SELECTION OPTIONS
==================================================

Treat visible selection options as fields/options.

For example:

[ ] Own
[ ] Rent

Return:

FIELD: Own
FIELD: Rent

==================================================
TABLES
==================================================

If the form contains a table, identify the table
structure separately.

However, do not convert normal fields into a table.

==================================================
SIGNATURE FIELDS
==================================================

If a signature field is physically present, include it.

For example:

FIELD: Applicant Signature
FIELD: Date

==================================================
DO NOT EXTRACT DECORATIVE TEXT
==================================================

Do not treat the following as fields unless they
clearly represent an actual field:

- company logos
- company names
- document titles
- decorative text
- instructions
- paragraphs
- legal disclaimers
- explanatory text

For example:

Home Loan Application

should not automatically become:

FIELD: Home Loan Application

However:

Loan Amount Requested ($)

is an actual field and must be extracted.

==================================================
PRESERVE DOCUMENT ORDER
==================================================

Keep fields in the same order in which they appear
on the document.

For example:

Loan Amount Requested ($)
Purpose of Loan
Preferred Repayment Term

must appear before:

SECTION: Applicant Information

and Applicant Information must appear before:

SECTION: Employment & Income

Do not rearrange fields alphabetically.

Do not move fields into sections where they do not
actually appear.

==================================================
MULTI-PAGE DOCUMENTS
==================================================

The OCR may contain multiple pages.

Analyze the COMPLETE OCR text.

Do not stop after the first page.

Extract fields from every page.

Preserve the order in which sections and fields
appear.

==================================================
OCR ERRORS
==================================================

OCR may contain errors.

Use the surrounding context to understand obvious
OCR mistakes.

For example:

"Loan Amount Requested ($)"

may appear as:

"Loan Amount Requested ($)"

or with minor OCR errors.

Correct obvious OCR errors when the intended field
is clear.

Do not invent fields.

If something cannot reliably be identified as a
field, do not create a field from it.

==================================================
DUPLICATE OCR INFORMATION
==================================================

OCR may contain repeated information.

Do not return the same field multiple times simply
because it appears more than once in OCR.

Return each actual form field once.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY the form structure.

Do NOT return JSON.

Do NOT provide explanations.

Do NOT provide field values.

Do NOT add Markdown tables.

Use exactly this format:

FIELD: [field name]

for fields before the first section.

Then:

SECTION: [actual section name]

FIELD: [field name]

FIELD: [field name]

Then the next section:

SECTION: [actual section name]

FIELD: [field name]

Continue until every actual field in the document
has been identified.

==================================================
EXAMPLE
==================================================

If the OCR contains:

Home Loan Application

Loan Requested

Loan Amount Requested ($)
Purpose of Loan
Preferred Repayment Term

New Purchase
Resale Property
Construction
Home Improvement

Applicant Information

Full Legal Name
Date of Birth
Street Address
City / Province / Postal Code
Phone Number
Email Address

Employment & Income

Employer / Organization Name
Job Title / Position
Gross Monthly Income ($)

Return:

FIELD: Loan Amount Requested ($)
FIELD: Purpose of Loan
FIELD: Preferred Repayment Term
FIELD: New Purchase
FIELD: Resale Property
FIELD: Construction
FIELD: Home Improvement

SECTION: Applicant Information

FIELD: Full Legal Name
FIELD: Date of Birth
FIELD: Street Address
FIELD: City / Province / Postal Code
FIELD: Phone Number
FIELD: Email Address

SECTION: Employment & Income

FIELD: Employer / Organization Name
FIELD: Job Title / Position
FIELD: Gross Monthly Income ($)

==================================================
CRITICAL RULES
==================================================

1. Analyze the complete document.

2. Extract every actual form field.

3. Do not ignore fields before the first section.

4. Fields before the first section must be returned
   before the first SECTION line.

5. Do not invent fields.

6. Do not use a predefined field list.

7. Preserve the original document order.

8. Preserve actual section names.

9. Extract checkbox and selection options that are
   physically present.

10. Extract signature fields that are physically
    present.

11. Extract fields from every page.

12. Do not duplicate fields.

13. Do not return field values.

14. Do not return JSON.

15. Do not provide explanations.

16. Return ONLY FIELD and SECTION lines.

==================================================
OCR DOCUMENT
==================================================

{ocr_text}
"""

    response = chat.invoke(prompt)

    return response.content
