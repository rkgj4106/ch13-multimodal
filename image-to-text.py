import base64
from pathlib import Path

import requests

# Read the image file
image_bytes = Path("animals.jpg").read_bytes()
image_base64 = base64.b64encode(image_bytes).decode("ascii")

# Send the image and question to Gemma 3
response = requests.post(
    "http://localhost:11435/api/chat",
    json={
        "model": "gemma3:12b",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Describe the entire image in English. "
                    "Explain the overall scene, the main people or objects, "
                    "the background, and their arrangement. "
                    "Do not speculate about details that are not clearly visible."
                ),
                "images": [image_base64]
            }
        ],
        "stream": False
    },
    timeout=600
)

response.raise_for_status()
print(response.json()["message"]["content"])