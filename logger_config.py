import logging
import os
import sys
from datetime import datetime


def configurar_logger(nome="aplicacao"):

    # Identificação única da execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Cria uma pasta para a execução
    diretorio_log = os.path.join("logs", timestamp)
    os.makedirs(diretorio_log, exist_ok=True)

    # Arquivo de log
    arquivo_log = os.path.join(diretorio_log, f"{nome}.log")

    logger = logging.getLogger(nome)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:

        formato = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(
            arquivo_log,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formato)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formato)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info("Diretório dos logs: %s", diretorio_log)

    return logger