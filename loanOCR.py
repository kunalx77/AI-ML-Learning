import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# streamlit ui

st.title("Indian Loan Form Information Extractor")

st.write(
    "Upload an Indian loan form image and extract " "the loan type, fields, and values."
)

# upload image

uploaded_file = st.file_uploader(
    "Upload an Indian loan form", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # open image

    image = Image.open(uploaded_file).convert("RGB")

    st.write(f"**Selected file:** {uploaded_file.name}")

    # display original image

    st.image(image, caption="Uploaded Loan Form", width=600)

    # extract information

    if st.button("Extract Information"):

        # preprocess image

        with st.spinner("Improving image quality..."):

            try:

                # convert image to grayscale

                gray_image = ImageOps.grayscale(image)

                # increase image size for better ocr

                width, height = gray_image.size

                scale = 2

                gray_image = gray_image.resize(
                    (width * scale, height * scale), Image.Resampling.LANCZOS
                )

                # improve contrast

                contrast_image = ImageEnhance.Contrast(gray_image).enhance(2.0)

                # sharpen image

                sharpened_image = contrast_image.filter(ImageFilter.SHARPEN)

                # apply additional sharpening

                sharpened_image = sharpened_image.filter(
                    ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
                )

            except Exception as e:

                st.error(f"Error while improving image: {e}")

                st.stop()

        # display processed image

        with st.expander("View Processed Image"):

            st.image(sharpened_image, caption="Processed Image", width=600)

        # ocr

        with st.spinner("Reading loan form..."):

            try:

                # first ocr pass

                text_1 = pytesseract.image_to_string(sharpened_image, config="--psm 6")

                # second ocr pass

                text_2 = pytesseract.image_to_string(sharpened_image, config="--psm 11")

                # third ocr pass

                text_3 = pytesseract.image_to_string(sharpened_image, config="--psm 3")

                # combine ocr results

                text = text_1 + "\n" + text_2 + "\n" + text_3

            except Exception as e:

                st.error(f"Error while reading image: {e}")

                st.stop()

        # show ocr text

        with st.expander("View OCR Text"):

            st.text(text)

        # check ocr result

        if not text.strip():

            st.warning("No text could be extracted from the image.")

            st.stop()

        # llm extraction

        with st.spinner("Identifying loan form and extracting fields..."):

            try:

                # create huggingface endpoint

                llm = HuggingFaceEndpoint(
                    repo_id="Qwen/Qwen2.5-7B-Instruct",
                    task="text-generation",
                    max_new_tokens=3000,
                    temperature=0.1,
                )

                # create chat model

                chat = ChatHuggingFace(llm=llm)

                # extraction prompt

                prompt = f"""
You are an expert Indian loan-document
classification and information extraction assistant.

Analyze the OCR text extracted from an Indian loan form.

Your tasks are:

1. Identify the loan form type.
2. Extract ALL fields present in the form.
3. Extract the value of every field.
4. Do not guess any information.

POSSIBLE LOAN TYPES:

- Personal Loan
- Home Loan
- Housing Loan
- Education Loan
- Vehicle Loan
- Car Loan
- Two-Wheeler Loan
- Business Loan
- MSME Loan
- Gold Loan
- Agricultural Loan
- Loan Against Property
- Consumer Loan
- Other

IMPORTANT RULES:

- Identify the loan type only from information present in the OCR text.
- If the loan type cannot be determined, return "Cannot Determine".
- Extract EVERY field present in the document.
- Preserve the original field name as much as possible.
- Do not create fields that are not present.
- Do not omit fields that are present.
- Do not guess missing information.
- If a field has a value, extract the exact value.
- If a field is empty, return "Empty/Not Filled".
- Preserve checkbox information when it appears in the OCR.
- Preserve Indian currency symbols such as ₹.
- Preserve dates, phone numbers, PAN-like text, Aadhaar-like text,
  addresses, names, numbers, and other values exactly as shown.
- Do not assume that an empty field has a value.
- Do not infer information from common loan-form patterns.
- Do not create placeholder fields.
- Do not extract placeholders.
- Only return the field name and its actual value.

RETURN EXACTLY THIS FORMAT:

Loan Form Type: ...

Fields:

1. Field: ...
   Value: ...

2. Field: ...
   Value: ...

3. Field: ...
   Value: ...

Continue for ALL fields present in the form.

OCR TEXT:

{text}
"""

                # invoke model

                response = chat.invoke(prompt)

                # get response content

                extracted_information = response.content

            except Exception as e:

                st.error(f"Error while extracting information: {e}")

                st.stop()

        # display result

        st.subheader("Extracted Loan Information")

        st.success("Information extracted successfully!")

        st.text(extracted_information)
