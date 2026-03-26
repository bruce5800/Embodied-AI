# Please install OpenAI SDK first: `pip3 install openai`
# pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-e12881e9ac7a4ab2a8a5a67ef21b083b",
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)