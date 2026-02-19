-- Orders breakdown by status
SELECT
    status,
    COUNT(*)            AS total_orders,
    SUM(amount)         AS total_revenue,
    AVG(amount)         AS avg_order_value,
    MIN(amount)         AS min_order,
    MAX(amount)         AS max_order
FROM transportation_orders
GROUP BY status
ORDER BY total_orders DESC;


-- Orders per shipping method
SELECT
    shipping_method,
    COUNT(*)        AS total_orders,
    AVG(amount)     AS avg_amount
FROM transportation_orders
GROUP BY shipping_method
ORDER BY total_orders DESC;


-- Top 10 customers by total spend
SELECT
    c.id,
    c.first_name || ' ' || c.last_name  AS customer_name,
    c.customer_type,
    COUNT(o.id)                         AS total_orders,
    SUM(o.amount)                       AS total_spent
FROM customers c
JOIN transportation_orders o ON o.customer_id = c.id
GROUP BY c.id, c.first_name, c.last_name, c.customer_type
ORDER BY total_spent DESC
LIMIT 10;
