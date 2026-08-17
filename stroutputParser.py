from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
parser = StrOutputParser()
chain = template_1 | model | parser | template_2 | model | parser
result = chain.invoke({"topic": "AI/ML TECHNOLOGY"})
print(result)
