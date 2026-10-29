# 문서 RAG + Gemma + 한국어 음성 웹 챗봇

질문 입력 → 문서 검색 → Gemma 답변 → 출처 표시 → 브라우저 음성 재생.
PDF/TXT/MD 파일을 웹에서 등록합니다. 스캔 PDF는 OCR 후 등록하세요.

## Windows 실행

압축을 풀고 `rag-voice-chat` 폴더에서 PowerShell로 실행합니다. Python 3.12 권장.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Ollama 서버에는 `gemma3:12b`와 `bge-m3`를 준비합니다. Kubernetes Ollama라면:

```powershell
kubectl exec deployment/ollama -- ollama pull gemma3:12b
kubectl exec deployment/ollama -- ollama pull bge-m3
kubectl port-forward svc/ollama 11435:11434
```

실제 Deployment/Service 이름이 다르면 바꾸고 필요 시 `-n <namespace>`를 추가하세요.
포트포워딩은 별도 창에서 계속 켜 둡니다. PC의 11434 충돌을 피하기 위해 11435를 사용합니다.
`.env`를 아래처럼 수정합니다.

```ini
OLLAMA_BASE_URL=http://localhost:11435
CHAT_MODEL=gemma3:12b
EMBED_MODEL=bge-m3
MIN_SCORE=0.35
```

PC 자체의 Ollama를 사용하면 URL은 `http://localhost:11434`입니다. 모델 준비는 `ollama pull <모델명>`으로 합니다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

http://localhost:8000 접속. 8000이 사용 중이면 `--port 8001`로 변경하세요.
가상환경 활성화 없이 실행하므로 PowerShell 실행 정책을 변경할 필요가 없습니다.

## 사용

1. 파일을 선택하고 **문서 등록**을 누릅니다. 임베딩 생성이 끝날 때까지 기다립니다.
2. 질문을 입력하고 **질문 보내기**를 누릅니다. Ctrl+Enter도 가능합니다.
3. 답변 아래 출처를 펼치면 검색한 문단과 PDF 페이지를 확인할 수 있습니다.
4. **답변 읽기** 버튼으로 음성을 재생하고 **읽기 중지**로 멈춥니다.
5. **답변 자동 읽기**를 켜면 답변 생성 후 자동 재생을 시도합니다.
6. **문서 RAG 사용**을 해제하면 일반 Gemma 답변을 받습니다.

질문은 요청마다 독립적입니다. 후속 질문에는 대상과 조건을 다시 적어주세요.
파일당 20MB, 최대 2,000개 문단. TXT/MD는 UTF-8/CP949 지원.
인터넷 검색, 마이크 입력, 이미지 분석은 포함하지 않습니다.

## 음성

브라우저 SpeechSynthesis로 답변을 읽습니다. 서버에서 MP3/WAV 파일을 생성하거나 다운로드하지 않습니다.
음성 종류와 품질, 오프라인 재생 가능 여부는 접속 PC/브라우저에 따라 달라집니다.
한국어 음성이 없으면 Windows 언어/음성 설정에서 한국어 음성을 설치하고 브라우저를 재시작하세요.
자동 재생이 제한되면 **답변 읽기** 버튼을 직접 누르세요. 서버에 스피커가 없어도 접속 PC에서 재생합니다.

## RAG 처리와 저장

문서를 약 900자 문단, 150자 겹침으로 나눠 `/api/embed`로 임베딩을 생성합니다.
질문과 코사인 유사도가 높은 상위 4개 문단을 `/api/chat`에 보내 답변을 생성합니다.
관련 자료를 찾지 못하면 일반 답변으로 자동 전환하지 않고 자료 부족을 알립니다.
검색 유사도는 정답 확률이 아닙니다. `MIN_SCORE`는 문서와 질문으로 평가해 조정하세요.
근거 중심 답변을 지시하지만 모델의 사실성과 출처 연결은 완전히 보장되지 않습니다.
화면의 출처는 검색된 문단이며, 모델이 정확히 인용했는지 확인할 수 있도록 표시합니다.

문단 텍스트·임베딩·파일명·페이지는 `data/rag.sqlite3`에 유지되며 원본 파일 자체는 저장하지 않습니다.
임베딩 생성이 전부 성공한 경우에만 저장합니다. 같은 파일과 모델의 중복 등록을 방지합니다.
임베딩 모델을 변경하면 이전 모델 문서는 검색에서 제외됩니다. 삭제 후 다시 등록하세요.
동일 모델 이름의 임베딩 차원이 변경되어도 삭제 후 재등록해야 합니다.
소규모 실습용으로, 검색 시 현재 모델의 문단을 메모리로 읽습니다. 대규모에는 별도 벡터 DB가 필요합니다.

## Linux / Docker / Kubernetes

Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Docker Desktop에서 호스트의 Ollama 연결 예:

```bash
docker build -t rag-voice-chat:1.0 .
docker run --rm -p 8000:8000 -v rag-voice-data:/data -e OLLAMA_BASE_URL=http://host.docker.internal:11434 rag-voice-chat:1.0
```

일반 Linux에서는 접근 가능한 Ollama 주소를 지정하세요.
Kubernetes 배포 시 같은 namespace의 Ollama는 `http://ollama:11434`, 다른 namespace는
`http://ollama.<namespace>.svc.cluster.local:11434`를 사용합니다. Pod에서 localhost는 해당 Pod 자신입니다.
환경 변수 `CHAT_MODEL`, `EMBED_MODEL`, `DATA_DIR=/data`를 설정하고 `/data`에는 PVC를 마운트하세요.
컨테이너 내부에만 저장하면 Pod 교체 시 자료가 사라질 수 있습니다. replica와 uvicorn worker는 각각 1로 사용하세요.
인증은 포함하지 않습니다. 기본 실행은 PC localhost용입니다. 공유 배포에는 접근 제한을 추가하세요.

## 오류와 검증

- 모델 없음: 프로그램이 연결하는 Ollama 서버에서 모델을 pull 하세요.
- 연결 오류: `.env` URL과 포트포워딩, Service/namespace를 확인하세요.
- 자료 검색 실패: 문서 등록 여부, 질문 구체성, 검색 임계값을 확인하세요.
- 첫 요청 지연: 모델 로딩과 GPU 메모리를 확인하세요. 두 모델이 번갈아 로딩될 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 Ollama 응답을 대체하여 등록·검색·근거 없음·삭제·실패 시 미저장·중복·모델 변경·입력 검증을 확인합니다.
실제 서버 연결, Gemma 답변 품질, PC의 한국어 음성은 해당 환경에서 확인해야 합니다.

공식 API: https://docs.ollama.com/api/embed / https://docs.ollama.com/api/chat
음성 API: https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis
