# Build:
#   docker build --no-cache -t ceemdan-model:v2 .
#
# CPU:
#   docker run --rm -it --entrypoint bash -v "${PWD}:/app" ceemdan-model:v2
#   python3 -u run_france.py
#
# Note: this image uses a clean Python base instead of modifying the
# preinstalled Python packages inside tensorflow/tensorflow.

FROM tensorflow/tensorflow:2.12.0

RUN python3 -m pip install --no-cache-dir --force-reinstall pip==22.3.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Dependências científicas compatíveis com TensorFlow 2.12
RUN python3 -m pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1

RUN python3 -m pip install --no-cache-dir --no-deps \
    pandas==1.5.3

RUN python3 -m pip install --no-cache-dir --no-deps \
    scikit-learn==1.2.2

RUN python3 -m pip install --no-cache-dir \
    Pillow==10.0.1

RUN python3 -m pip install --no-cache-dir \
    matplotlib==3.7.3

# Instalar PyEMD e EWT sem modificar os arquivos-fonte
RUN python3 -m pip install --no-cache-dir \
    EMD-signal==1.5.2 \
    ewtpy==0.2

RUN python3 -m pip install --no-cache-dir \
    python-dateutil==2.8.2 pytz==2023.3 \
    joblib==1.4.2 threadpoolctl==3.5.0

COPY . .

CMD ["python", "-u", "run_france.py"]
