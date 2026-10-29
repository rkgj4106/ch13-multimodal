# Gemma + Wan: real text-to-video motion generation

English script -> Gemma motion prompts -> Wan-generated temporal frames -> edited MP4.
This project replaces the text-card renderer with an actual text-to-video diffusion model.
It does not use still-image pan/zoom as a substitute for video synthesis.
The code, user interface, prompts, examples, captions and optional narration are English.

## Your server configuration

| Component | Configuration |
|---|---|
| Node | sict |
| Namespace | default |
| Existing GPUs | 2 x A100 80GB PCIe |
| Existing Ollama | ollama.default.svc.cluster.local:11434, one GPU |
| Motion application | one GPU, separate from the Ollama Pod |
| Planner | gemma3:12b in the existing Ollama service |
| Video model | Wan-AI/Wan2.1-T2V-1.3B-Diffusers |
| Output | 832 x 480, 16 fps, 49 or 81 generated frames per shot |

Each GPU has its own memory. This implementation does not combine them into a single 160GB
device and does not use distributed inference. Wan 1.3B is the initial model for a manageable
first deployment. Quality varies by prompt and motion; the project does not promise a specific
rendering time on your server. Increasing steps can improve quality while increasing compute.

## What changes from the previous project

- engine.py calls Diffusers WanPipeline with num_frames, not Pillow text-card rendering.
- The model generates subjects, backgrounds and temporal motion from English prompts.
- Gemma does not generate pixels: it writes the motion prompts.
- FFmpeg adds captions after video generation and joins separate shots.
- This is a new Deployment, Service, PV and PVC; previous videos are not automatically migrated.
- Namespaces and the Ollama address are default, matching your current setup.

## Files

| File | Purpose |
|---|---|
| engine.py | Gemma JSON planning, Wan model loading/inference, captions/audio and FFmpeg |
| app.py | HTTP API, durable job status, one GPU job worker |
| cli.py | Script or reviewed storyboard command-line generation |
| index.html | English web UI with job progress and downloads |
| sample_script.txt | Imagined historical Mongolian riding scene |
| example_storyboard.json | One manually prepared motion prompt, for bypassing the planner |
| requirements.txt | Pinned video stack plus API dependencies |
| Dockerfile | PyTorch 2.8 / CUDA 12.8 runtime, FFmpeg, fonts and app |
| k8s.yaml | Persistent storage, GPU Deployment and internal Service |

## 1. Check the existing services

Run on the Mongolia server, or in Windows with kubectl selecting that cluster:

```bash
kubectl get nodes
kubectl get svc ollama -n default
kubectl exec -n default deployment/ollama -- ollama list
kubectl describe node sict
```

Make sure gemma3:12b exists. If missing:

```bash
kubectl exec -n default deployment/ollama -- ollama pull gemma3:12b
```

In describe node, the Allocated resources section must leave at least one nvidia.com/gpu free.
The node's allocatable GPU count can remain 2 even when devices are already assigned to Pods.
If Ollama requests both GPUs, reduce its request to one before deploying this app.
This manifest does not change Ollama or assign fixed GPU indexes; the device plugin allocates devices.

On the Ubuntu server, check storage space:

```bash
df -h /var/lib
```

The first request downloads substantial model weights from Hugging Face. Keep ample free storage
for the cache, downloaded weights and generated clips. The 100Gi hostPath capacity is a declaration,
not a reservation or enforced disk quota.

## 2. Build the image

In the extracted project directory on Windows, with Docker Desktop using Linux containers:

```powershell
docker login
docker build --platform linux/amd64 -t yklee2002/gemma-motion-video:1.0 .
docker push yklee2002/gemma-motion-video:1.0
```

The CUDA/PyTorch base image is large, so building, pushing and the first node pull take longer
than the previous CPU-only application. The build validates video-library imports but does not
download model weights and does not require a GPU. If using a private repository, configure a
registry pull secret in default and reference it in the Pod spec.

## 3. Deploy the complete manifest

```powershell
kubectl apply -f k8s.yaml
kubectl get pv gemma-motion-pv
kubectl get pvc gemma-motion-data -n default
kubectl get pods -n default -l app=gemma-motion-video -w
```

Apply the entire YAML, including the PV: deploying only the PVC/Deployment leaves the Pod Pending.
It uses the node's /var/lib/gemma-motion-video directory and mounts it at /data.
The existing gemma-video app can remain installed; this project uses different resource names.

Check the GPU visible inside the new Pod:

```powershell
kubectl exec -n default deployment/gemma-motion-video -- python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Expected: CUDA True, device count 1, and an A100 name.
No additional toolkit/runtime changes should be needed after the NVIDIA device plugin is working.

## 4. Open the app

From Windows, automatically select an available local port:

```powershell
kubectl port-forward -n default --address 127.0.0.1 svc/gemma-motion-video :8000
```

Use the port printed in Forwarding from 127.0.0.1:PORT -> 8000.
Keep this terminal open and visit http://127.0.0.1:PORT in your browser.

## 5. First test

1. Select Load example.
2. Choose 1 shot, 49 frames, 30 steps and Silent.
3. Select Generate motion video.
4. Wait through PLANNING, LOADING_MODEL, GENERATING and ENCODING.
5. Download/play video.mp4. The subject and camera should have model-generated motion.

The API starts before loading Wan. Running/Ready means the API is available, not that weights are
already loaded. The first job downloads and loads weights, which may take a long time.
Readiness/liveness checks do not call Gemma or run video inference.

Follow progress in another terminal:

```powershell
kubectl logs -n default deployment/gemma-motion-video -f
```

After the first successful test, use 81 frames and multiple shots. At 16 fps, 49 frames are about
3.06 seconds and 81 frames are about 5.06 seconds. Total duration is the sum of generated shots,
with small encoding differences. Multiple minutes of finished footage require many generated clips
and substantial processing time; the web UI caps a request at six shots.

## Audio, captions and consistency

- Silent mode uses local model inference after weights are downloaded, with a silent audio track.
- English narration uses online gTTS and sends narration text to Google's speech service.
- If narration exceeds the generated clip duration, FFmpeg slows the moving clip to fit the audio.
  It does not freeze a final frame. To preserve the original motion speed, use Silent or shorter narration.
- Captions use approximate character-based timing, not forced word alignment.
- Separate text-to-video shots are generated independently. Repeated prompt descriptions do not guarantee
  consistent faces, costumes, geography or identity between shots.
- Historical scenes are imagined reconstructions and may contain inaccuracies or visual artifacts.
- Lip synchronization, uploaded-image animation, reference-character consistency, background music and
  automatic source fact-checking are not included in this first implementation.

## Persistence and operation

| Path in Pod | Actual location on sict |
|---|---|
| /data/huggingface | /var/lib/gemma-motion-video/huggingface |
| /data/jobs | /var/lib/gemma-motion-video/jobs |

The model cache and outputs persist across Pod replacement on the same node. The hostPath volume
does not replicate to other nodes. Retain does not automatically erase data on resource deletion.
Intermediate raw clips, captions, audio and encoded shots are retained; there is no cleanup policy.
Monitor disk space. Run one Pod and one Uvicorn worker; the Python worker serializes GPU jobs.
At most three active/queued jobs are accepted. Queued jobs resume after a restart; interrupted
running jobs are marked failed and require resubmission. The existing cached model files are reused.

The local web app has no authentication. Use localhost port forwarding or a controlled lab network.
No paid video-generation API is used. Internet access is needed for image builds/model downloads
and for optional online TTS. After weights are cached, silent inference can run without Hugging Face
downloads by setting HF_HUB_OFFLINE=1 if the cache is complete.

## CLI in the GPU container

The image includes cli.py. To avoid a second process competing for the same GPU, do not run it
inside the web Pod while a web job is active. For a dedicated standalone GPU container:

```bash
python cli.py --script sample_script.txt --scenes 1 --frames 49 --steps 30 --voice none --output /data/cli-output
```

Bypass Gemma with a reviewed scene plan:

```bash
python cli.py --storyboard example_storyboard.json --frames 49 --steps 30 --output /data/cli-output
```

Input scripts are read as text. The application never executes input Python code or model output.

## Troubleshooting

| Symptom | Check |
|---|---|
| Pending / Insufficient nvidia.com/gpu | Existing Pod allocations; Ollama must leave one GPU free |
| Pending / unbound PVC | Apply the PV and PVC; check names, namespace and claimRef |
| Name or service not known | OLLAMA_BASE_URL and DNS; this YAML uses ollama.default |
| Missing model in Ollama | Pull gemma3:12b in the existing Ollama container |
| CUDA unavailable | Resource limits, NVIDIA device plugin and container runtime |
| CUDA out of memory | Start with one shot/49 frames; ensure the Pod has its own visible GPU |
| OOMKilled | Host RAM limit/available memory; inspect Pod events |
| LOADING_MODEL takes a long time | Hugging Face connectivity, download logs and free cache disk space |
| gTTS failure | Try Silent to isolate external speech-service connectivity |
| Poor motion or anatomy | Simplify action, describe camera movement, change seed or increase steps |

Use new image tags for updates and change the image in k8s.yaml. Configuration changes in the
Deployment Pod template trigger a restart. This manifest does not delete the previous app or data.

## Validation limits

The implementation was checked against the pinned Diffusers 0.35.1 pipeline API and VAE tiling interface.
Python syntax, YAML references and application/job behavior are tested locally. Actual FFmpeg
postprocessing is tested with synthetic moving test footage; that is not a Wan-generated demo.
This authoring environment has no CUDA GPU, so live Wan inference, generated-scene quality, the
full CUDA image build, online TTS and the user's cluster deployment were not verified here.
No generated sample is included that could be mistaken for a successful Wan run.

Official references:
- https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- https://huggingface.co/docs/diffusers/api/pipelines/wan
- https://docs.ollama.com/capabilities/structured-outputs
