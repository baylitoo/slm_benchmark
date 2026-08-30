import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
template = {"store": "verbatim-string", "total": "number", "currency": ["USD", "EUR", "GBP"]}

resp = client.chat.completions.create(
    model="hf.co/numind/NuExtract3-GGUF:Q4_K_M",
    temperature=0.2,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Yesterday I bought apples and coffee at Trader Joe's for $12.40."}
    ]}],
    extra_body={"chat_template_kwargs": {"template": json.dumps(template), "enable_thinking": False}},
)
print(resp.choices[0].message.content)