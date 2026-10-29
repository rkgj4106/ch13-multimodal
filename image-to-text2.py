import base64
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11435"  # Ollama server address and port
MODEL = "gemma3:12b"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "animals.jpg"
MAX_HISTORY_TURNS = 6  # Keep the six most recent question-and-answer pairs.
INSTRUCTIONS = (
    "Answer the user's questions in English while examining the same attached image. "
    "Describe positions as left, right, top, bottom, or center from the viewer's perspective. "
    "When asked about the location of a specific object, also describe its position "
    "relative to nearby objects. "
    "Refer to previous questions and answers, but base your answers on the actual image. "
    "Do not speculate about anything that is not visible or is uncertain."
)


class ImageChat:
    def __init__(self, image_path):
        image_path = Path(image_path).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_bytes = image_path.read_bytes()
        if not image_bytes:
            raise ValueError("The image file is empty.")
        self.image_base64 = base64.b64encode(image_bytes).decode("ascii")
        self.history = []

    def clear(self):
        self.history.clear()

    def ask(self, question):
        if not question.strip():
            raise ValueError("Please enter a question.")
        user_message = {"role": "user", "content": question}

        # The API does not automatically retain conversations between requests,
        # so include the conversation history in every request.
        messages = [dict(message) for message in self.history] + [dict(user_message)]

        # Include the image once per request without accumulating image messages.
        messages[0]["content"] = INSTRUCTIONS + "\n\nQuestion: " + messages[0]["content"]
        messages[0]["images"] = [self.image_base64]

        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={"model": MODEL, "messages": messages, "stream": False},
            timeout=(10, 600),
        )
        if not response.ok:
            raise RuntimeError(f"Ollama error ({response.status_code}): {response.text}")
        data = response.json()
        if data.get("error"):
            raise RuntimeError(data["error"])
        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")

        # Add only successfully completed question-and-answer pairs to the history.
        self.history.extend([user_message, {"role": "assistant", "content": answer}])
        self.history = self.history[-MAX_HISTORY_TURNS * 2:]
        return answer


def main():
    raw_path = input(
        "Image path (Enter: animals.jpg in the same folder): "
    ).strip()
    image_path = Path(raw_path.strip('\"').strip("'")) if raw_path else DEFAULT_IMAGE

    try:
        chat = ImageChat(image_path)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    print(f"\nImage: {image_path}\nModel: {MODEL}")
    print("Ask multiple questions about the same image. q: quit / clear: reset conversation")

    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                print("Exiting.")
                break
            if question.lower() == "clear":
                chat.clear()
                print("Conversation cleared. Continue asking about the same image.")
                continue
            if not question:
                continue

            print("Analyzing the image...", flush=True)
            print("\n[Answer]\n" + chat.ask(question))

        except requests.exceptions.ConnectionError:
            print(
                f"Connection failed: Check the address {OLLAMA_URL} "
                "and your port forwarding."
            )
        except requests.exceptions.Timeout:
            print("Response timed out: Check Ollama's status and try again.")
        except (ValueError, RuntimeError, requests.exceptions.RequestException) as exc:
            print(f"Error: {exc}")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()