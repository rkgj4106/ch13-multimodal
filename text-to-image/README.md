# 설명을 그림으로: SDXL 웹 프로그램

한국어 설명 → Ollama/Gemma 영어 변환 → SDXL 이미지 생성 → PNG 표시와 다운로드.
Gemma는 프롬프트 변환을 담당하며 실제 이미지는 별도의 SDXL 모델이 생성합니다.
영어 설명을 직접 입력하면 Ollama 없이도 실행할 수 있습니다.

## 먼저 확인: GPU 한 장과 기존 Ollama

NVIDIA GPU, 호환 드라이버, NVIDIA Container Toolkit, Kubernetes NVIDIA Device Plugin이 필요합니다.
RTX 5090를 고려해 PyTorch 2.7.1 / CUDA 12.8 wheel을 설치하도록 Dockerfile을 구성했습니다.

```powershell
kubectl get pods -A -o wide
kubectl describe node
```

일반 NVIDIA Device Plugin의 `nvidia.com/gpu: 1`은 GPU 하나를 독점 할당합니다.
Ollama가 단일 GPU를 이미 요청한 경우 이 Pod는 `Insufficient nvidia.com/gpu`로 Pending이 될 수 있습니다.
VRAM이 남아 있어도 Kubernetes 자원 예약 때문에 배치되지 않을 수 있습니다.
한글 변환과 이미지 생성 Pod를 동시에 실행하려면 GPU를 따로 배정하거나 Device Plugin의 time-slicing 공유 설정이 필요합니다.
공유 설정은 Helm/GPU Operator/DaemonSet 등 현재 설치 방식에 맞게 적용해야 하므로 이 YAML은 클러스터 설정을 변경하지 않습니다.
Time-slicing은 VRAM 분리나 메모리 보장을 제공하지 않습니다. 두 모델의 총 메모리 사용량을 확인하세요.

공유 없이 먼저 시험하려면 기존 Ollama replica 수를 기록한 뒤, 테스트 동안만 GPU 사용 Pod를 중지하여 GPU 자원을 확보합니다.
YAML의 `TRANSLATE_DEFAULT`를 `false`로 바꾸고 영어 프롬프트를 입력하세요.
완료 후 이미지 생성 Deployment를 중지하고 기존 Ollama replica 수를 복원합니다.
다른 앱이 Ollama를 사용 중이면 중지 시 영향을 받으므로 현재 운영 상태를 확인하고 수행하세요.

Gemma 변환에는 `keep_alive: 0`을 지정하여 요청 뒤 모델 언로드를 요청합니다.
모델 언로드가 Kubernetes GPU 자원 예약을 해제하지는 않습니다.
이미지 생성 Pod는 SDXL 모델을 GPU에 유지합니다. 공유 상태에서 메모리가 부족하면
`CPU_OFFLOAD=true`로 일부 모델을 CPU로 옮길 수 있으나 속도는 느려질 수 있습니다.
GPU 요청을 단순히 삭제하는 것으로 GPU 공유가 설정되지는 않습니다.

## 빌드 / 배포

압축을 풀고 `text-to-image` 폴더에서 실제 계정 이름으로 바꾸어 실행하세요.

```powershell
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_ID/text-to-image:1.0 .
docker push YOUR_DOCKERHUB_ID/text-to-image:1.0
```

PyTorch/CUDA 라이브러리 때문에 Docker 이미지 크기와 빌드 시간이 큽니다.
모델 가중치는 Docker 이미지에 포함하지 않으며 첫 생성 작업 때 다운로드합니다.
`k8s.yaml`에서 `image`를 실제 업로드한 이미지 이름으로 바꾸세요.
기본 Ollama 주소는 `http://ollama.default.svc.cluster.local:11434`입니다.
Service나 namespace가 다르면 `OLLAMA_BASE_URL`을 수정하세요. 해당 Ollama 서버에 `gemma3:12b`가 필요합니다.

```powershell
kubectl get storageclass
kubectl get runtimeclass
```

기본 StorageClass가 없으면 PVC에 실제 `storageClassName`을 지정하세요. 해당 프로비저너 또는 정적 PV도 필요합니다.
기존 Ollama에 `runtimeClassName: nvidia`가 설정되어 있다면 이 YAML에도 필요에 따라 주석을 해제하세요.
기본 NVIDIA 런타임을 사용하는 클러스터는 별도 RuntimeClass 지정이 필요하지 않을 수 있습니다.

```powershell
kubectl apply -f k8s.yaml
kubectl get pods,pvc,svc -n default
kubectl logs -n default deployment/text-to-image -f
kubectl port-forward -n default svc/text-to-image 8002:8000
```

http://localhost:8002 에서 접속합니다. 내부망에서는 `http://<GPU서버IP>:30801`로 접속할 수 있습니다.
NodePort와 방화벽 접근이 가능해야 합니다. 30801이 이미 사용 중이면 다른 사용하지 않는 포트로 변경하세요.
새 버전은 새 이미지 태그로 build/push한 뒤 YAML image 태그를 바꾸어 apply하세요.

## 사용과 저장

1. 장면·색상·구도·화풍을 입력합니다.
2. 한글이면 Gemma 영어 변환을 켭니다. 영어 입력이면 꺼도 됩니다.
3. 크기, 생성 단계, Seed를 선택하고 이미지 생성을 누릅니다.
4. 완료 후 PNG 다운로드를 누릅니다. 실제 사용한 프롬프트도 표시됩니다.

예: 몽골 초원 위에 하얀 게르 세 개, 노을과 멀리 달리는 말, 따뜻한 수채화.
영어 예: A small friendly robot reading a book in a bright classroom, watercolor illustration.

기본 크기 1024×1024, 30 steps. SDXL base fp16 모델을 사용합니다.
첫 생성은 모델 다운로드/로딩 시간이 포함됩니다. 서버에서 huggingface.co와 모델 다운로드 CDN 접근이 필요합니다.
GPU 작업은 하나씩 처리하며 실행 중 포함 최대 8개 작업을 접수합니다.
PVC 30Gi에 모델 캐시와 결과 PNG/작업 정보를 저장합니다. 작업량에 따라 용량을 늘리세요.
자동으로 생성 이미지를 삭제하지 않으므로 장기 사용 시 디스크 정리 정책이 필요합니다.
Pod replica와 uvicorn worker는 각각 1로 유지하세요.
Pod 재시작 시 미완료 작업은 실패로 처리하며 자동 재실행하지 않습니다.
화면은 현재 탭의 작업 ID를 기억하며 새로고침 후 조회할 수 있습니다.

Gemma 변환은 의미를 바꿀 수 있으니 실제 프롬프트를 확인하세요.
모델 토큰 한도를 넘는 긴 프롬프트는 일부 잘릴 수 있으며 화면에 경고를 표시합니다.
정확한 문자·복잡한 관계도·정밀한 수량이나 배치는 잘 표현하지 못할 수 있습니다.
이 프로그램은 새 이미지 생성용입니다. 이미지 편집·RAG·음성 기능은 포함하지 않습니다.
모델을 다른 아키텍처로 바꿀 경우 모델 이름 외에 파이프라인 코드도 변경해야 합니다.
실습용이며 인증 기능은 포함하지 않습니다. 외부 공개 시 접근 제한이 필요합니다.

## 문제 확인

```powershell
kubectl describe pod -n default -l app=text-to-image
kubectl describe pvc -n default text-to-image-data
kubectl logs -n default deployment/text-to-image
```

- GPU 부족으로 Pending: 기존 GPU 예약과 공유 설정 확인
- PVC Pending: StorageClass/프로비저너 또는 정적 PV 확인
- CUDA 없음: NVIDIA 런타임/Device Plugin 확인
- GPU 메모리 부족: Ollama 등 다른 프로세스 점유, 크기, CPU_OFFLOAD 확인
- 다운로드 실패: 서버 네트워크와 모델 저장소 접근, PVC 여유 공간 확인
- 영어 변환 실패: Ollama Service/모델 확인. 영어 입력+변환 해제로 분리 테스트

`/health`는 웹 프로세스 응답 상태입니다. GPU 모델의 로딩 성공 여부나 GPU 연산 성공을 보장하지 않습니다.
모델 상태는 응답의 `model_state`로 확인하고 실제 생성 작업으로 검증하세요.

## 검증 범위

API 테스트는 모델을 대체하여 작업 등록·조회·입력 검증·PNG 응답·재시작 처리를 확인합니다.
실제 GPU 모델 다운로드, SDXL 생성, 드라이버 호환성, 클러스터 배포는 대상 서버에서 확인해야 합니다.
이 코드 작성 환경에서는 실제 GPU 모델 실행을 검증하지 않았습니다.

공식 문서:
- https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- https://pytorch.org/blog/pytorch-2-7/
- https://pytorch.org/get-started/previous-versions/
- https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html
