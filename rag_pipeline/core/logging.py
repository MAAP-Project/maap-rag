import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_file = "logs/maap_rag_pipeline.log", level = logging.INFO):
    fmt = logging.Formatter(
        "%(asctime)sPST %(levelname)s %(name)s - %(message)s"
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called twice
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
