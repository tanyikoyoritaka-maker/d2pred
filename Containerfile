FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/d2pred_final
COPY requirements-lock.txt .
RUN python3.11 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-lock.txt
COPY . .
CMD ["python3.11", "scripts/audit_final_preexecution.py"]
