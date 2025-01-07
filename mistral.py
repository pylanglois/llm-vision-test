import os

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

_client = None


def init_client():
    global _client
    if _client is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        _client = Mistral(api_key=api_key)
    return _client


def text_completion(system_prompt, user_prompt, base64_images):
    client = init_client()
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content":
                [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ] + [
                    {"type": "image_url",
                     "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    for base64_image in base64_images]
        }
    ]
    # model = "pixtral-12b-2409"
    model = "pixtral-large-latest"
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        response_format={
            "type": "json_object",
        }
    )

    # Print the content of the response
    return chat_response.choices[0].message.content
