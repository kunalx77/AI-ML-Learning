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


""")

print(response.content)
