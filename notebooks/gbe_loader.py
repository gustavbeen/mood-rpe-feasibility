"""Load GBE participant records using the local data configuration."""

from pathlib import Path
import tomllib

from scipy.io import loadmat

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.local.toml"
MAT_FILENAME = "Rutledge_GBE_risk_data_TOD.mat"


def load_gbe():
    """Read the configured MAT file and return its subjData records."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "Copy config.example.toml to config.local.toml and set data_dir."
        )

    with CONFIG_PATH.open("rb") as config_file:
        settings = tomllib.load(config_file)

    folder = settings.get("data_dir")
    if not isinstance(folder, str) or not folder.strip():
        raise ValueError("Set data_dir in config.local.toml to the GBE data folder.")

    data_path = CONFIG_PATH.parent / Path(folder).expanduser() / MAT_FILENAME
    if not data_path.is_file():
        raise FileNotFoundError(f"Could not find {MAT_FILENAME} in configured data_dir.")

    return loadmat(data_path, simplify_cells=True)["subjData"]
