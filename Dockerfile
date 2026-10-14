FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD --checksum=sha256:936ab0904982dc98f625b412c55089be64d70ef77ba8a4122bcde0298ad076f8 \
    https://huggingface.co/Xenova/segformer-b4-finetuned-ade-512-512/resolve/0f92f0b465567a1a1f004ec33110992b6158edbc/onnx/model.onnx \
    /opt/models/segformer-b4.onnx

COPY renderer/ renderer/
COPY bench/ bench/
COPY textures/ textures/
COPY tests/ tests/
COPY pytest.ini .

EXPOSE 8100

CMD ["python", "-m", "uvicorn", "bench.app:app", "--host", "0.0.0.0", "--port", "8100"]
