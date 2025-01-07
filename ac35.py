import json
import os

import boto3
import dotenv

dotenv.load_dotenv()

_bedrock_runtime_client = None


def init_bedrock_runtime_client():
    global _bedrock_runtime_client
    if _bedrock_runtime_client is None:
        _bedrock_runtime_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv("AWS_REGION"),
        )
    return _bedrock_runtime_client


def text_completion(system_prompt, user_prompt, base64_images):
    client = init_bedrock_runtime_client()
    # model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    # model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

    payload = {
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content":
                    [
                        {
                            "type": "text",
                            "text": user_prompt,
                        }
                    ] + [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                        for base64_image in base64_images
                    ]
            }
        ],
        "max_tokens": 1000,
        "anthropic_version": "bedrock-2023-05-31"
    }

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        body=json.dumps(payload)
    )

    output_binary = response["body"].read()
    output_json = json.loads(output_binary)
    output = output_json["content"][0]["text"]
    return output
