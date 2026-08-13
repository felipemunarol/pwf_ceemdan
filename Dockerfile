# Commands: 
#   - docker build -t ceemdan-model:v1 .
#   - executa em cpu: docker run -it -v "${PWD}:/app" ceemdan-model:v1
#   - executa em cpu (alternativa direto do container): docker run -it --entrypoint bash -v "${PWD}:/app" ceemdan-model:v1
#   (após entrar executar python3 run_frande.py)
#   - executa em gpu: docker run -it --gpus all -v "${PWD}:/app" ceemdan-model:v1

FROM tensorflow/tensorflow:2.3.0

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN python3 -m pip install --upgrade pip==21.3.1

# =========================
# Wheels locais
# =========================

COPY pandas-0.25.3-cp36-cp36m-manylinux1_x86_64.whl /tmp/
COPY scikit_learn-0.24.2-cp36-cp36m-manylinux1_x86_64.whl /tmp/
COPY matplotlib-3.3.4-cp36-cp36m-manylinux1_x86_64.whl /tmp/

RUN pip3 install --no-index --no-cache-dir --no-deps \
    /tmp/pandas-0.25.3-cp36-cp36m-manylinux1_x86_64.whl

RUN pip3 install --no-cache-dir \
    python-dateutil==2.8.2 \
    pytz==2021.1 \
    six==1.15.0

RUN pip3 install --no-index --no-cache-dir --no-deps \
    /tmp/scikit_learn-0.24.2-cp36-cp36m-manylinux1_x86_64.whl

RUN pip3 install --no-cache-dir \
    joblib==1.0.1 \
    threadpoolctl==2.1.0

RUN pip3 install --no-cache-dir \
    Pillow==8.4.0

RUN pip3 install --no-cache-dir \
    scipy==1.4.1

RUN pip3 install --no-index --no-cache-dir --no-deps \
    /tmp/matplotlib-3.3.4-cp36-cp36m-manylinux1_x86_64.whl


# =========================
# EMD / EWT
# =========================

RUN pip3 install --no-cache-dir \
    EMD-signal==0.2.10

RUN pip3 install --no-cache-dir \
    ewtpy==0.2


# =========================
# Dependências do emd
# =========================

RUN pip3 install --no-cache-dir --no-deps numpydoc==1.1.0
RUN pip3 install --no-cache-dir --no-deps tabulate==0.8.10
RUN pip3 install --no-cache-dir --no-deps PyYAML==6.0.1
RUN pip3 install --no-cache-dir --no-deps dcor==0.5.3
RUN pip3 install --no-cache-dir --no-deps sparse==0.13.0
RUN pip3 install --no-cache-dir --no-deps pathos==0.2.8


# =========================
# EMD
# =========================

RUN pip3 install --no-cache-dir --no-deps \
    emd==0.7.0


# =========================
# Código
# =========================

COPY . .

CMD ["python3", "run_france.py"]