from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import json
from pydantic import BaseModel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

res = ChatHuggingFace(llm=llm)


class Schema(BaseModel):
    summary: str
    sentiment: str


result = res.invoke("""
Return ONLY valid JSON:

{
    "summary": "...",
    "sentiment": "positive, negative, or neutral"
}

Do not add any other fields.

Addie had lived in Happyville since she was born. Next week, however, Addie
and her family were moving over 1,000 miles away to Washington. Addie despised
the idea of moving for many reasons. She was sad to be leaving her best friend.
She had played on the soccer team for two years and didn’t want to leave her team.
She would not be sleeping in her bedroom, which she loved and had decorated all by
herself. She just hated the whole thing.

Addie’s dad had gotten a new job and said it would be good for the entire
family. Her mother told Addie that there would be a lot of new things to do and
people to meet. Her brother was too young to understand.

The whole situation was worse because they were moving on Addie’s birthday.
She was going to turn 11 and wanted to spend the day with her friends.

One morning Addie woke up and decided to try a new approach. She would
make a plan about how this could actually be a good thing. She took pictures
of everything familiar to her and made a list of things she could try in the
new town. She would join the soccer team and introduce herself to the kids
at school to make friends with them.
""")
content = result.content
data = json.loads(content)
validated_result = Schema.model_validate(data)
print(validated_result)
print(data)
