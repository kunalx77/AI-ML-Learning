from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", task="text-generation")
model = ChatHuggingFace(llm=llm)
template_1 = PromptTemplate(
    template="write a detailed report on {topic}", input_variables=["topic"]
)

template_2 = PromptTemplate(
    template="write a 7 line summary on {text}",
    input_variables=["text"],
)

prompt1 = template_1.invoke({"topic": "AI/ML Technology"})
result = model.invoke(prompt1)

prompt2 = template_2.invoke({"text": result.content})
result_2 = model.invoke(prompt2)
print(result_2.content)
