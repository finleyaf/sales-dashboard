import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

CSV = repo_root() / "data" / "online_retail_clean.csv"
DB_DIR = repo_root() / "db"
DB_PATH = DB_DIR / "retail.db"
TABLE = "online_retail"

def clean_colnames(df):
    df.columns = [c.strip().lower().replace(" ", "_").replace(".", "") for c in df.columns]
    return df

def main():
    # Ensure the database directory exists
    DB_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV)
    df = clean_colnames(df)

    # Force int for customer_id
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype("Int64")

    # Convert invoice_date to datetime
    if "invoicedate" in df.columns:
        df["invoicedate"] = pd.to_datetime(df["invoicedate"], dayfirst=True, errors="coerce")

    # Create database engine and load data
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    df.to_sql(TABLE, engine, if_exists="replace", index=False)

    # Create indexes for faster queries
    with engine.connect() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_customer_id ON {TABLE} (customer_id);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_invoice_date ON {TABLE} (invoicedate);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_product ON {TABLE} (description);"))

    print("Loaded", len(df), "rows into", DB_PATH)

if __name__ == "__main__":
    main()