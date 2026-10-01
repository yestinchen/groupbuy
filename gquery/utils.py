from pathlib import Path
import logging


def setup_logging(algorithm_name: str):
    log_dir = Path("log") / algorithm_name
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"{algorithm_name}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
