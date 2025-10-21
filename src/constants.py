from pathlib import Path
import os

# Allow override via environment variable for VM deployment
work_dir = Path(os.environ.get("WORK_DIR", "/workdir"))
data_dir = Path(os.environ.get("DATA_DIR", work_dir / "data"))
configs_dir = work_dir / "configs"
soccernet_dir = Path(os.environ.get("SOCCERNET_DIR", data_dir / "soccernet"))
