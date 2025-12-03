
-- 1) Monthly revenue
SELECT strftime('%Y-%m', invoicedate) AS year_month,
        SUM(totalprice) AS total_revenue,
        COUNT(DISTINCT invoice) AS orders
FROM online_retail
WHERE invoicedate IS NOT NULL
GROUP BY year_month
ORDER BY year_month;
-- "SELECT strftime('%Y-%m', invoicedate) AS year_month, SUM(totalprice) AS total_revenue, COUNT(DISTINCT invoice) AS orders FROM online_retail WHERE invoicedate IS NOT NULL GROUP BY year_month ORDER BY year_month;"

-- 2) Top 10 products by revenue
SELECT description AS product,
        SUM(totalprice) AS total_revenue,
        SUM(quantity) AS total_quantity
FROM online_retail
GROUP BY description
ORDER BY total_revenue DESC
LIMIT 10;

-- 3) Top 20 customers by lifetime value
SELECT customer_id,
        SUM(totalprice) AS lifetime_value,
        COUNT(DISTINCT invoice) AS orders
FROM online_retail
WHERE customer_id IS NOT NULL
GROUP BY customer_id
ORDER BY lifetime_value DESC
LIMIT 20;

-- 4) Revenue by country (top 10)
SELECT country,
        SUM(totalprice) AS total_revenue,
        COUNT(DISTINCT customer_id) AS customers
FROM online_retail
GROUP BY country
ORDER BY total_revenue DESC
LIMIT 10;

-- 5) Cohort: monthly new customers
WITH first_orders AS (
    SELECT customer_id, MIN(strftime('%Y-%m', invoicedate)) AS first_month
    FROM online_retail
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
)
SELECT first_month, COUNT(*) AS new_customers
FROM first_orders
GROUP BY first_month
ORDER BY first_month;
-- "WITH first_orders AS (SELECT customer_id, MIN(strftime('%Y-%m', invoicedate)) AS first_month FROM online_retail WHERE customer_id IS NOT NULL GROUP BY customer_id) SELECT first_month, COUNT(*) AS new_customers FROM first_orders GROUP BY first_month ORDER BY first_month;"