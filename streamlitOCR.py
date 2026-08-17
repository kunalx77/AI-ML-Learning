import streamlit as st
import pytesseract
import io
from PIL import Image
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from docx import Document
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# docx processing


def process_docx(file_bytes):

    document = Document(io.BytesIO(file_bytes))

    text = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            text.append(paragraph.text)

    return "\n".join(text)


# pdf processing


def process_pdf(file_bytes):

    pages = convert_from_bytes(file_bytes, dpi=300)

    all_text = []

    for page in pages:

        text = pytesseract.image_to_string(page)

        all_text.append(text)

    return "\n".join(all_text), pages


st.title("Document Information Extractor")

st.write("Upload a document and extract important information.")


# file uploader

uploaded_file = st.file_uploader("Upload a document", type=None)


if uploaded_file is not None:

    file_name = uploaded_file.name.lower()
    file_type = uploaded_file.type

    st.write(f"**Selected file:** {uploaded_file.name}")

    st.write(f"**File type:** {file_type}")

    if st.button("Extract Information"):

        # step1 read document

        with st.spinner("Reading document..."):

            try:

                # image files

                if file_name.endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")
                ):

                    image = Image.open(uploaded_file)

                    st.image(image, caption="Uploaded Document", width=500)

                    text = pytesseract.image_to_string(image)

                # pdf file

                elif file_name.endswith(".pdf"):

                    text, pages = process_pdf(uploaded_file.getvalue())

                    for i, page in enumerate(pages):

                        st.image(page, caption=f"Page {i + 1}", width=500)

                # docx file

                elif file_name.endswith(".docx"):

                    text = process_docx(uploaded_file.getvalue())

                # txt file

                elif file_name.endswith(".txt"):

                    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

                # unsupported file

                else:

                    st.error(f"Unsupported file type: {uploaded_file.name}")

                    st.stop()

            except Exception as e:

                st.error(f"Error while reading document: {e}")

                st.stop()

        # ocr text

        with st.expander("View OCR Text"):

            st.text(text)

        # check text

        if not text.strip():

            st.warning("No text could be extracted from the document.")

            st.stop()

        # extract infon

        with st.spinner("Extracting information..."):

            try:

                llm = HuggingFaceEndpoint(
                    repo_id="Qwen/Qwen2.5-7B-Instruct",
                    task="text-generation",
                )

                chat = ChatHuggingFace(llm=llm)

                prompt = f"""
You are an expert international document classification and information extraction assistant.

Analyze the document text and extract ONLY these 8 fields:

1. Document Type
2. Country
3. Name
4. Gender
5. Date of Birth
6. Document Number
7. Expiry Date
8. Document Issue Date

The document can be from ANY country.

Possible document types include:

- Passport
- Driving Licence
- PAN Card
- National ID Card
- Government ID
- Residence Permit
- Visa
- Tax Identification Card
- Voter ID
- Work Permit
- Other

IMPORTANT:

- Always return all 8 fields.
- If a field is not present, return "Cannot Determine".
- If a field is unclear because of OCR, return "Cannot Determine".
- Never leave a field blank.
- Never omit a field.
- Do NOT guess any value.

<<<<<<< HEAD
=======
Gender may be represented by labels such as:
- Gender
- Sex
- G
- S
- Male / M
- Female / F
- Other / O

If any of these explicitly indicate the person's gender/sex, extract the value.

Do NOT infer gender from the person's name.
Do NOT guess gender.
If the gender/sex cannot be clearly determined, return "Cannot Determine".
>>>>>>> 55053e9 (Add output parser examples and OCR updates)

Gender rules:
- Extract Gender ONLY if it is explicitly stated in the document.
- Valid values are: Male, Female, Other.
- Do NOT infer gender from the person's name.
- Do NOT infer gender from pronouns.
- Do NOT guess based on the document type or country.
- If gender is unclear, return "Cannot Determine".

Follow this exact format:

Document Type: ...
Country: ...
Name: ...
Gender: ...
Date of Birth: ...
Document Number: ...
Expiry Date: ...
Document Issue Date: ...

Document text:
{text}
"""

                response = chat.invoke(prompt)

            except Exception as e:

                st.error(f"Error while extracting information: {e}")

                st.stop()

        # show result

        st.subheader("Extracted Information")

        st.success("Information extracted successfully!")

        st.text(response.content)
