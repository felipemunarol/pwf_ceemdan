# Build:
#   docker build --no-cache -t ceemdan-model:v1 .
#
# CPU:
#   docker run -it --entrypoint bash -v "${PWD}:/app" ceemdan-model:v1
#
# GPU:
#   docker run -it --gpus all --entrypoint bash -v "${PWD}:/app" ceemdan-model:v1


FROM tensorflow/tensorflow:2.12.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Preservar o TensorFlow e as dependências da imagem original.
# Instalar apenas as bibliotecas adicionais do projeto.
RUN python3 -m pip install --no-cache-dir \
    pandas==1.5.3 \
    scikit-learn==1.2.2 \
    matplotlib==3.7.3 \
    Pillow==10.0.1 \
    EMD-signal==1.5.2 \
    ewtpy==0.2

COPY . .

CMD ["python3", "-u", "run_france.py"]

