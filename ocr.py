import pytesseract
from PIL import Image
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# step1 loading doc
image = Image.open("documents/sample pan.png")

# step2 extract text using tesseract

text = pytesseract.image_to_string(image)

print(" OCR OUTPUT :")
print(text)
print("END")

# step3 connect to hf llm

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)
chat = ChatHuggingFace(llm=llm)

# step4 prompt

prompt = f"""
You are a document information extraction assistant.

Extract the following information from the document text:

- Name
- Date of Birth
- Document Number

Document text:
{text}

Return only these three fields in this format:

Name: ...
Date of Birth: ...
Document Number: ...
"""

# step5 ocr text to llm

response = chat.invoke(prompt)

# step6 display infon
print(" EXTRACTED INFORMATION : ")
print(response.content)
print("END")
