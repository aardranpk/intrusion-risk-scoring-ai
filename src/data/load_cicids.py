import pandas as pd
from pathlib import Path

# Always resolve paths relative to THIS file
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PARQUET_FILE = DATA_DIR / "Portscan-Friday-no-metadata.parquet"

def load_data():
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {PARQUET_FILE}")

    df = pd.read_parquet(PARQUET_FILE)

    df.columns = df.columns.str.strip()
    df = df.dropna()

    return df


if __name__ == "__main__":
    df = load_data()
    print("Loaded data shape:", df.shape)
    print(df.head())
