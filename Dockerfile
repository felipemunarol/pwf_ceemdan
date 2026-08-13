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

COPY requirements_docker.txt .

# NÃO inclui o tensorflow no requirements
RUN python3 -m pip install --no-cache-dir -r requirements_docker.txt

COPY . .

CMD ["python3", "run_france.py"]