FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD --checksum=sha256:936ab0904982dc98f625b412c55089be64d70ef77ba8a4122bcde0298ad076f8 \
    https://huggingface.co/Xenova/segformer-b4-finetuned-ade-512-512/resolve/0f92f0b465567a1a1f004ec33110992b6158edbc/onnx/model.onnx \
    /opt/models/segformer-b4.onnx

ADD --checksum=sha256:719a498cf5b3fe9be9f01ee513e13d3915f9028aa4f23dfd30eaaa0a17143159 \
    https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/32266dda3cf1d5ea1cdccd386de1eb3355d0f0b9/weights/edge_sam_3x_encoder.onnx \
    /opt/models/edge_sam_3x_encoder.onnx
ADD --checksum=sha256:83a2174d54571596913dcb7455d021e713623c3dca30a31c8c41ab98c9fb0863 \
    https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/32266dda3cf1d5ea1cdccd386de1eb3355d0f0b9/weights/edge_sam_3x_decoder.onnx \
    /opt/models/edge_sam_3x_decoder.onnx

COPY renderer/ renderer/
COPY bench/ bench/
COPY textures/ textures/
COPY tests/ tests/
COPY pytest.ini .

EXPOSE 8100

CMD ["python", "-m", "uvicorn", "bench.app:app", "--host", "0.0.0.0", "--port", "8100"]
