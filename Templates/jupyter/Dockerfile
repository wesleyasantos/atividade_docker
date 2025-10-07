FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV JUPYTER_TOKEN=mlops \
    JUPYTER_ENABLE_LAB=yes

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=${JUPYTER_TOKEN}", "--NotebookApp.notebook_dir=/notebooks"]


