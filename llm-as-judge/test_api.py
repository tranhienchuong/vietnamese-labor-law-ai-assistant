import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("THESPARK_API_KEY"),
    base_url=os.getenv("THESPARK_BASE_URL"),
)

response = client.chat.completions.create(
    model=os.getenv("THESPARK_MODEL", "gpt-5.4"),
    messages=[
        {"role": "system", "content": "Bạn là trợ lý tiếng Việt, trả lời ngắn gọn."},
        {"role": "user", "content": "Xin chào, bạn đang hoạt động không?"}
    ],
)

print(response.choices[0].message.content)