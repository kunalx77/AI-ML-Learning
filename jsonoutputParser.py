from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", task="text-generation")
model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
template = PromptTemplate(
    template="give me the name,age,city of 5 main characters of GTA V \n {format_instruction} ",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)
chain = template | model | parser

result = chain.invoke({})
print(result["name"])
