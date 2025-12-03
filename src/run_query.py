import pandas as pd
from sqlalchemy import create_engine
from load_sqlite import repo_root

CSV = repo_root() / "data" / "online_retail_clean.csv"
DB_DIR = repo_root() / "db"
DB_PATH = DB_DIR / "retail.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

def sql_to_csv(query, output_csv):
    df = pd.read_sql_query(query, engine)
    df.to_csv(output_csv, index=False)
    print(f"Saved query results to {output_csv}")

if __name__ == "__main__":
    q = "SELECT strftime('%Y-%m', invoicedate) AS year_month, SUM(totalprice) AS total_revenue, COUNT(DISTINCT invoice) AS orders FROM online_retail WHERE invoicedate IS NOT NULL GROUP BY year_month ORDER BY year_month;"
    output_csv = repo_root() / "outputs" / "monthly_revenue.csv"
    sql_to_csv(q, output_csv)