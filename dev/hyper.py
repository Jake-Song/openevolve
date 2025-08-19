import os
import openai

system_content = "You are a gourmet. Be descriptive and helpful."
user_content = "Tell me about Chinese hotpot"

client = openai.OpenAI(
    api_key=os.getenv("HYPERBOLIC_API_KEY"),
    base_url="https://api.hyperbolic.xyz/v1",
    )

chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ],
    temperature=0.7,
    max_tokens=1024,
)

response = chat_completion.choices[0].message.content
print("Response:\n", response)