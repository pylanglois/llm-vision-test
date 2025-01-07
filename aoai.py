import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

_client = None
_model = os.getenv("OPENAI_MODEL")


def init_client():
    global _client
    if _client is None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        _client = AzureOpenAI(
            azure_endpoint=api_base,
            api_key=api_key,
            api_version=api_version,
        )
    return _client


def text_completion(system_prompt, user_prompt, base64_images):
    client = init_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            [{"type": "text", "text": user_prompt}] +
            [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                for base64_image in base64_images
            ]}
    ]
    response = client.chat.completions.create(
        model=_model,
        messages=messages,
    )
    return response.choices[0].message.content
