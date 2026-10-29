import base64
from pathlib import Path

import requests

# Forward port 11435 on your PC to port 11434 of Ollama in Kubernetes.
OLLAMA_URL = "http://127.0.0.1:11435"
MODEL = "gemma3:12b"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "ingress.png"
MAX_HISTORY_TURNS = 6

GUIDE = """Analyze the attached system architecture diagram and answer in English.
Analysis rules:
- Base your analysis on the component names, boxes, lines, arrows, and connection labels actually visible in the image.
- Determine component roles from their displayed descriptions and names. Mark roles inferred from general knowledge as [Inferred].
- For each connection, identify the source component, destination component, arrow direction, and label.
- Do not assume every connection represents data flow. Distinguish data transfer, request/response, and control relationships.
  If there is insufficient evidence to identify the relationship type, mark it as [Cannot Determine].
- Do not guess the direction or connected components when arrowheads are unclear or lines overlap.
- Do not invent components, protocols, processing sequences, or data formats that are not shown in the image.
- Describe visible parallel branches, merges, and bidirectional connections, but explain execution order only when supported by the diagram.
- Mark unreadable names and unclear connections as [Illegible] or [Cannot Determine].
- Refer to previous answers and follow-up questions, but prioritize evidence from the image if they conflict."""

OVERVIEW_QUESTION = """Explain the components and data flows in this system architecture diagram.
Use the following order:
1. Overall architecture: Summarize the system structure visible in the diagram.
2. Components: Provide each name, role, and supporting evidence, or indicate whether the role is inferred.
3. Connections: Use the format 'Source → Destination / Connection label / Relationship type'.
4. Data flows: Explain the paths along which data moves through confirmed connections.
5. Points requiring clarification: Identify unclear text, arrows, connections, and data meanings.
Do not present details absent from the diagram as established facts."""


class DiagramChat:
    def __init__(self, image_path):
        image_path = Path(image_path).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"Diagram image not found: {image_path}")
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Please provide a PNG, JPG, or WEBP image file.")
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

        # Include the image and recent conversation in every request.
        messages = [dict(message) for message in self.history] + [dict(user_message)]
        messages[0]["content"] = GUIDE + "\n\nQuestion:\n" + messages[0]["content"]
        messages[0]["images"] = [self.image_base64]

        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 4096},
            },
            timeout=(10, 600),
        )
        if not response.ok:
            raise RuntimeError(
                f"Ollama error ({response.status_code}): {response.text}"
            )
        data = response.json()
        if data.get("error"):
            raise RuntimeError(data["error"])
        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")
        if data.get("done_reason") == "length":
            answer += (
                "\n\n[Warning: The response was truncated "
                "because the output length limit was reached.]"
            )

        # Do not record failed requests in the conversation history.
        self.history.extend([user_message, {"role": "assistant", "content": answer}])
        self.history = self.history[-MAX_HISTORY_TURNS * 2:]
        return answer


def show_answer(chat, question):
    try:
        print("Analyzing the diagram...", flush=True)
        print("\n[Analysis Results]\n" + chat.ask(question))
    except requests.exceptions.ConnectionError:
        print(
            f"Connection failed: Check the address {OLLAMA_URL} "
            "and your port forwarding."
        )
    except requests.exceptions.Timeout:
        print("Response timed out: Check Ollama's status and try again.")
    except (ValueError, RuntimeError, requests.exceptions.RequestException) as exc:
        print(f"Error: {exc}")


def main():
    raw_path = input(
        "Diagram image path (Enter: ingress.png in the same folder): "
    ).strip()
    image_path = Path(raw_path.strip('\"').strip("'")) if raw_path else DEFAULT_IMAGE

    try:
        chat = DiagramChat(image_path)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    print(f"\nImage: {image_path}\nModel: {MODEL}\nOllama: {OLLAMA_URL}")
    show_answer(chat, OVERVIEW_QUESTION)

    print(
        "\nAsk follow-up questions / overview: full analysis / "
        "clear: reset conversation / q: quit"
    )

    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                break
            if question.lower() == "clear":
                chat.clear()
                print(
                    "Conversation cleared. "
                    "Continue asking about the same diagram."
                )
                continue
            if not question:
                continue
            show_answer(
                chat,
                OVERVIEW_QUESTION if question.lower() == "overview" else question
            )
        except (EOFError, KeyboardInterrupt):
            break

    print("\nExiting.")


if __name__ == "__main__":
    main()