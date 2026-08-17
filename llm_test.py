from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=100,
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke("""
Extract the following information from the document text:
- Name
- Date of Birth
- Document Number

Document text:
INCOME TAX DEPARTMENT
Permanent Account Number Card
ELWPM8089J
Name
RAHUL MISHRA
Father's Name
SATENDRA MISHRA
30/01/1997
""")

print(response.content)
