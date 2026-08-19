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
        provider="auto",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=2000,
        temperature=0.0,
    )

    return ChatHuggingFace(llm=llm)


# extract information from id


def extract_id_information(ocr_text):

    chat = create_model()

    prompt = f"""
You are an identity document information
extraction assistant.

Extract information that is actually present
in the identity document.

Do not guess.

Do not invent information.

Return information using exactly:

FIELD: Full Name
VALUE: ...

FIELD: Date of Birth
VALUE: ...

FIELD: Address
VALUE: ...

FIELD: Identification Number
VALUE: ...

Only return fields that are actually present.

OCR TEXT:

{ocr_text}
"""

    response = chat.invoke(prompt)

    return response.content


# parse id information


def parse_id_information(extraction):

    information = {}

    current_field = None

    for line in extraction.splitlines():

        line = line.strip()

        if line.startswith("FIELD:"):

            current_field = line[len("FIELD:") :].strip()

        elif line.startswith("VALUE:"):

            value = line[len("VALUE:") :].strip()

            if current_field:

                information[current_field] = value

                current_field = None

    return information


# match id information to loan form fields


def match_id_to_form(id_information, form_sections):

    chat = create_model()

    form_fields = []

    for section_name, fields in form_sections:

        for field in fields:

            form_fields.append(field)

    prompt = f"""
You are matching information extracted from
a government identity document to fields in a
loan application form.

ID INFORMATION:

{id_information}

LOAN FORM FIELDS:

{form_fields}

Your job is to determine which ID information
belongs to which loan form field.

Only match information when the relationship
is clear.

Do not guess.

Do not invent values.

If there is no suitable ID value for a form
field, leave it empty.

Return ONLY this format:

FIELD: Full Legal Name
VALUE: John Smith

FIELD: Date of Birth
VALUE: 1995-05-10

FIELD: Street Address
VALUE: Ahmedabad

FIELD: Email Address
VALUE:

Every loan form field MUST appear exactly once.

If no ID information matches a field:

FIELD: Email Address
VALUE:

Do not add fields that are not present in the
loan form.

ID INFORMATION:

{id_information}

LOAN FORM FIELDS:

{form_fields}
"""

    response = chat.invoke(prompt)

    return response.content


# parse matched fields


def parse_matched_fields(extraction):

    matched = {}

    current_field = None

    for line in extraction.splitlines():

        line = line.strip()

        if line.startswith("FIELD:"):

            current_field = line[len("FIELD:") :].strip()

        elif line.startswith("VALUE:"):

            value = line[len("VALUE:") :].strip()

            if current_field:

                matched[current_field] = value

                current_field = None

    return matched


# extract filled loan form information


def extract_filled_form_information(ocr_text, form_sections):

    form_structure = ""

    for section_name, fields in form_sections:

        form_structure += f"\nSECTION: {section_name}\n"

        for field in fields:

            form_structure += f"FIELD: {field}\n"

    chat = create_model()

    prompt = f"""
You are extracting values from a filled loan application.

The OCR text below was generated from the actual filled
loan application.

The OCR preserves the approximate layout of the document.

Your job is to match values in the OCR to the backend
loan form fields.

==================================================
BACKEND FORM FIELDS
==================================================

{form_structure}

==================================================
OCR TEXT
==================================================

{ocr_text}

==================================================
RULES
==================================================

1. Return ONLY fields that have an actual value.

2. Do not invent values.

3. Do not guess.

4. Do not return empty fields.

5. Match a value to the closest corresponding field.

6. The value may appear on the line immediately below
   the field label.

7. The value may also appear on the same line as the
   field label.

8. Use the layout of the OCR to determine which value
   belongs to which field.

9. Preserve the value exactly as it appears in the OCR
   whenever possible.

10. The fields before the first SECTION are also real
    fields and must be extracted.

==================================================
EXAMPLE
==================================================

OCR:

Loan Amount Requested ($)                    Purpose of Loan
12,50,000                                    New family car

Preferred Repayment Term
5 years

Full Legal Name                              Date of Birth
Rahul Mehta                                  14/06/1994

Return:

FIELD: Loan Amount Requested ($)
VALUE: 12,50,000

FIELD: Purpose of Loan
VALUE: New family car

FIELD: Preferred Repayment Term
VALUE: 5 years

FIELD: Full Legal Name
VALUE: Rahul Mehta

FIELD: Date of Birth
VALUE: 14/06/1994

==================================================
SECOND EXAMPLE
==================================================

OCR:

Phone Number                                  Email Address
98765 42130                                   rahul.mehta94@mail.com

Return:

FIELD: Phone Number
VALUE: 98765 42130

FIELD: Email Address
VALUE: rahul.mehta94@mail.com

==================================================
THIRD EXAMPLE
==================================================

OCR:

Vehicle Make / Model
Maruti Suzuki Brezza

Vehicle Year
2026

Dealer / Seller Name
Shree Motors, Bharuch

Return:

FIELD: Vehicle Make / Model
VALUE: Maruti Suzuki Brezza

FIELD: Vehicle Year
VALUE: 2026

FIELD: Dealer / Seller Name
VALUE: Shree Motors, Bharuch

==================================================
IMPORTANT
==================================================

Do NOT return fields that are blank.

Do NOT return:

FIELD: Some Field
VALUE: Empty/Not Filled

Do NOT return:

FIELD: Some Field
VALUE: Cannot Determine

Only return fields for which the OCR contains a
recognizable value.

==================================================
CHECKBOXES
==================================================

The uploaded form may contain checkbox options.

You MUST identify checkbox state from the OCR.

ONLY return a checkbox if there is clear evidence that
the checkbox is CHECKED or SELECTED.

NEVER return unchecked checkbox options.

NEVER return every checkbox in a group.

For example, if the form contains:

New Vehicle
Used Vehicle
Private Sale
Dealer Purchase

and the OCR clearly indicates that only New Vehicle
is selected, return ONLY:

FIELD: New Vehicle
VALUE: New Vehicle
STATUS: Selected

Do NOT return:

FIELD: Used Vehicle
VALUE: Used Vehicle

Do NOT return:

FIELD: Private Sale
VALUE: Private Sale

Do NOT return:

FIELD: Dealer Purchase
VALUE: Dealer Purchase

Do NOT return:

STATUS: Not Selected

If the OCR does NOT provide enough information to
determine which checkbox is selected, DO NOT return
any checkbox from that group.

Never assume the first option is selected.

Never assume the last option is selected.

Never select a checkbox based only on its position.

Only return a checkbox when the OCR provides clear
evidence that it is checked.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY:

FIELD: field name
VALUE: value

Do not return explanations.

Do not return JSON.

Do not return markdown.

==================================================
BACKEND FORM
==================================================

{form_structure}

==================================================
OCR
==================================================

{ocr_text}
"""

    try:

        response = chat.invoke(prompt)

        return response.content

    except Exception as e:

        raise Exception(f"error extracting filled form information: {e}")
