FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY static ./static
ENV DATA_DIR=/data HF_HOME=/models PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
