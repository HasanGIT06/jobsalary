from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent.parent
INPUT_FILE = ROOT_DIR / "Dataset" / "B.csv"
INGESTED_DIR = BASE_DIR / "ingested"
OUTPUT_FILE = INGESTED_DIR / "B.csv"

def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    assert not df.empty, "Dataset is empty"
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data ingested: {INPUT_FILE} → {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_data()