import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from datetime import date
from typing import Tuple, cast

from data_prep import repo_root

# Setup

st.set_page_config(page_title="Sales Dashboard", layout="wide", initial_sidebar_state="expanded")

DB_DIR = repo_root() / "db"
DB_PATH = DB_DIR / "retail.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

@st.cache_data
def load_data(query):
    return pd.read_sql(query, engine)

st.title("Sales & Customer Analysis Dashboard")
st.markdown("Interactive dashboard powered by Python, SQL and Streamlit")

# Sidebar filters

st.sidebar.header("Filters")

date_query = """
SELECT 
    MIN(invoicedate) AS min_date,
    MAX(invoicedate) AS max_date 
FROM online_retail;
"""
date_range = pd.read_sql(date_query, engine)
min_date = pd.to_datetime(date_range["min_date"].iloc[0])
max_date = pd.to_datetime(date_range["max_date"].iloc[0])

result = st.sidebar.date_input(
    "Invoice date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Normalize to (start_date, end_date)
if isinstance(result, tuple):
    if len(result) == 2:
        start_date, end_date = result
    else:
        # Empty tuple: default to full range
        start_date, end_date = min_date.date(), max_date.date()
else:
    # Single date: use the same value for both ends
    d = cast(date, result)
    start_date, end_date = d, d

product_search = st.sidebar.text_input("Search product")

# Load filtered data

query = f"""
SELECT *
FROM online_retail
WHERE invoicedate BETWEEN '{start_date}' AND '{end_date}'
"""
data = pd.read_sql(query, engine)

if product_search:
    data = data[data["description"].str.contains(product_search, case=False, na=False)]

# KPI 


total_revenue = data["totalprice"].sum()
total_orders = data["invoice"].nunique()
unique_customers = data["customer_id"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"£{total_revenue:,.0f}")
col2.metric("🧾 Orders", f"{total_orders:,}")
col3.metric("👥 Unique Customers", f"{unique_customers:,}")

# Monthly revenue

st.subheader("Monthly Revenue Trend")

monthly_query = """
SELECT strftime('%Y-%m', invoicedate) AS year_month,
        SUM(totalprice) AS total_revenue,
        COUNT(DISTINCT invoice) AS orders
FROM online_retail
WHERE invoicedate IS NOT NULL
GROUP BY year_month
ORDER BY year_month;
"""
monthly_data = load_data(monthly_query)

fig1 = px.line(monthly_data, x="year_month", y="total_revenue", title="Monthly Revenue", markers=True)

fig1.update_layout(xaxis_title="Year-Month", yaxis_title="Total Revenue (£)")
st.plotly_chart(fig1, use_container_width=True)

# Top products

st.subheader("Top 10 Products by Revenue")

top_products_query = """
SELECT description AS product,
        SUM(totalprice) AS total_revenue,
        SUM(quantity) AS total_quantity
FROM online_retail
GROUP BY description
ORDER BY total_revenue DESC
LIMIT 10;
"""

top_products = load_data(top_products_query)

fig2 = px.bar(
    top_products,
    x="product",
    y="total_revenue",
    title="Top Products by Revenue"
)
fig2.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig2, use_container_width=True)

# Customer lifetime value

st.subheader("Top 20 Customers by Lifetime Value")

customer_lifetime_query = """
SELECT 
    customer_id,
    SUM(totalprice) AS lifetime_value,
    COUNT(DISTINCT invoice) AS visits
FROM online_retail
WHERE customer_id IS NOT NULL
GROUP BY customer_id
ORDER BY lifetime_value DESC
LIMIT 20;
"""

customer_lifetime = load_data(customer_lifetime_query)

fig3 = px.scatter(
    customer_lifetime,
    x="visits",
    y="lifetime_value",
    title="Customer Value vs Frequency",
    size="lifetime_value",
    hover_data=["customer_id"]
)
st.plotly_chart(fig3, use_container_width=True)

# Country revenue

st.subheader("Top 10 countries by revenue")

country_query = """
SELECT country,
        SUM(totalprice) AS total_revenue,
        COUNT(DISTINCT customer_id) AS customers
FROM online_retail
GROUP BY country
ORDER BY total_revenue DESC
LIMIT 10;
"""

country_data = load_data(country_query)

fig4 = px.bar(
    country_data,
    x="country",
    y="total_revenue",
    title="Top Countries by Revenue"
)
fig4.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig4, use_container_width=True)

# Footer

st.markdown("---")
st.markdown("""
Built with **Streamlit**, **SQLite**, **Python**, and **Plotly**  
by *Finley Ashton Foreman* — [GitHub Repo](https://github.com/finleyaf/sales-dashboard)
""")
