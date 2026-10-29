FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data/jobs HF_HOME=/data/huggingface HF_HUB_DISABLE_XET=1 \
    OLLAMA_BASE_URL=http://ollama.default.svc.cluster.local:11434 \
    LLM_MODEL=gemma3:12b VIDEO_MODEL=Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
# Validate imports during build. No model weights or GPU are required here.
RUN python -c "import torch; from diffusers import WanPipeline, AutoencoderKLWan; from transformers import UMT5EncoderModel; print(torch.__version__)"
COPY app.py engine.py cli.py index.html sample_script.txt example_storyboard.json ./
RUN mkdir -p /data/jobs /data/huggingface
EXPOSE 8000
CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1"]
