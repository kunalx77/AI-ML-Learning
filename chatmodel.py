from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)
history = []

model = ChatHuggingFace(llm=llm)
while True:
    user_input = input("YOU : ")
    history.append(user_input)
    if user_input == "exit":
        break
    res = model.invoke(history)
    history.append(res.content)
    print("A.I. :", res.content)
print(history)
