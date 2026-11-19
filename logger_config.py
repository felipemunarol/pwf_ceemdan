
import logging
import os
from datetime import datetime

def configurar_logger(nome="aplicacao"):
    # Diretório para armazenar os logs
    os.makedirs("logs", exist_ok=True)

    # Nome do arquivo com data e hora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_log = f"logs/{nome}_{timestamp}.log"

    logger = logging.getLogger(nome)
    logger.setLevel(logging.DEBUG)

    # Evita adicionar handlers duplicados
    if logger.handlers:
        return logger

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Arquivo
    file_handler = logging.FileHandler(
        arquivo_log,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formato)

    # Terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formato)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
