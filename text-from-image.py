import base64
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11435"
MODEL = "gemma3:12b"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BASE_DIR / "주주명부.jpg"
OUTPUT_DIR = BASE_DIR / "output"

PROMPT = """Extract the text and table structures from the attached document image.
Follow these rules:
1. Transcribe the visible content exactly as written. Do not summarize or translate it.
2. Output titles, paragraphs, lists, notes, and tables in their original reading order.
3. Label tables as [Table 1], [Table 2], etc., and format them as Markdown tables.
4. Preserve row order, column order, headers, numbers, dates, units, and symbols.
5. Leave empty cells blank. Mark unreadable characters or cells as [Illegible].
6. Do not invent or add content or values that are not visible in the image.
7. Markdown cannot fully represent merged cells or multi-level headers.
   If necessary, repeat headers in the corresponding columns and describe
   the visible merged-cell ranges under [Structure Notes] below the table.
   If a range is uncertain, explicitly state that it is uncertain.
8. For tables without headers, use placeholder headers such as 'Column 1'
   and 'Column 2'. State under [Structure Notes] that the original table
   has no headers.
9. If there are no tables, output only the document text.
10. Do not include an introduction, an explanation of the extraction method,
    or a code block enclosing the entire output.
Output only the document content and any necessary [Structure Notes]."""


def extract_document(image_path):
    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Document image not found: {image_path}")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("Please provide a JPG, PNG, or WEBP image file.")
    image_bytes = image_path.read_bytes()
    if not image_bytes:
        raise ValueError("The image file is empty.")

    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        json={
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": PROMPT,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 8192},
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
    text = data.get("message", {}).get("content", "").strip()
    if not text:
        raise RuntimeError("The extraction result is empty.")
    return text, data.get("done_reason") == "length"


def save_result(image_path, text, truncated):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"{Path(image_path).stem}_extracted_{stamp}.txt"
    if truncated:
        text = (
            "[Warning: The extraction result was truncated "
            "because the output length limit was reached.]\n\n"
            + text
        )

    # Create a new file for each run to preserve previous results.
    with output_path.open("x", encoding="utf-8") as output:
        output.write(text + "\n")
    return output_path


def main():
    raw_path = input(
        "Document image path (Enter: 주주명부.jpg in the same folder): "
    ).strip()
    image_path = Path(raw_path.strip('\"').strip("'")) if raw_path else DEFAULT_IMAGE

    try:
        print(
            f"Extracting text and tables... Model: {MODEL}",
            flush=True
        )
        text, truncated = extract_document(image_path)
        print("\n[Extraction Result]\n" + text)

        if truncated:
            print(
                "\nWarning: The output was truncated. "
                "Split the document into smaller sections or adjust num_predict."
            )

        output_path = save_result(image_path, text, truncated)
        print(f"\nUTF-8 text saved: {output_path}")

    except requests.exceptions.ConnectionError:
        print(
            f"Connection failed: Check the address {OLLAMA_URL} "
            "and your port forwarding."
        )
        raise SystemExit(1)

    except requests.exceptions.Timeout:
        print("Response timed out: Check Ollama's status.")
        raise SystemExit(1)

    except (OSError, ValueError, RuntimeError, requests.exceptions.RequestException) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()