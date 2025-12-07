# 📊 Sales & Customer Analytics Dashboard  
### End-to-end data analytics project using Python, SQL, SQLite, Streamlit & Plotly

This project demonstrates a full analytics workflow from **raw data → cleaned dataset → SQL database → insights → interactive dashboard**.  
It uses the **UCI Online Retail Dataset**, a real e-commerce transaction dataset widely used in analytics interviews.

The goal was to showcase:
- SQL for business analysis  
- Python data cleaning + feature engineering  
- An end-to-end data pipeline  
- Interactive dashboard development  
- Business-ready insights  

---

## Project Highlights

✔ Loaded & cleaned 540k+ transaction records  
✔ Built a reproducible SQLite analytical database  
✔ Wrote SQL queries for revenue, customer value, and product analysis  
✔ Built an interactive dashboard with Streamlit + Plotly  
✔ Created monthly revenue trends, top products, and customer value visuals  
✔ Exported insights for reporting + interview discussion

---

## Technologies Used

**Data Engineering:** Pandas, SQLite, SQLAlchemy  
**Analytics:** SQL, Python  
**Visualisation:** Matplotlib, Plotly  
**Dashboard:** Streamlit  
**Tools:** Jupyter Notebook, VS Code, GitHub  

---

## Data Preparation

Data cleaning steps included:
- Removing cancelled orders / negative quantities  
- Parsing invoice dates
- Removing missing Customer IDs  
- Normalising product descriptions  
- Feature engineering: month, week, day of week  

---

## SQL Database

All cleaned data is loaded into a **SQLite database** using:

```bash
python src/load_sqlite.py
```


The script:
- Normalises column names
- Converts invoice date to datetime 
- Creates indexes
- Saves into db/retail.db

---

## Key Analyses

Full SQL exploration is in:
notebooks/02-sql.ipynb

---

## Interactive Dashboard

Run locally:
```bash
streamlit run src/app_streamlit.py
```

Features:
- Date range filter
- Product search
- Monthly revenue trend
- Top 10 products by revenue
- Top 20 customers by lifetime value
- Revenue by country

---

## Insights

Summary:
- Monthly revenue shows a sudden growth spike starting January 2010 which is consistent until most recent data.
- Top 3 products are easily indentified with a steady decline on revenue threreafter.
- Less frequent customers with larger order values are more common in highest lifetime value customers.
- Overwhelming bias for UK ordering shows shifting focus into other regions could increase overall revenue.

A full summary is available in outputs/insights.pdf

---

## Running the project

1. Clone the repo:
```bash
git clone https://github.com/finleyaf/sales-dashboard.git
cd sales-dashboard
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Build the database:
```bash
python src/load_sqlite.py
```

5. Launch dashboard:
```bash
streamlit run src/app_streamlit.py
```

---

## Dataset

UCI Online Retail Dataset
Kaggle link: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci?resource=download

---

## Contact

Finley Ashton Foreman
Email: fashtonforeman@gmail.com
GitHub: https://github.com/finleyaf
LinkedIn: https://www.linkedin.com/in/finleyashtonforeman/

---

## License

This project is licensed under the MIT License