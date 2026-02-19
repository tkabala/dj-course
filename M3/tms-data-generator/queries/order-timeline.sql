-- Number of events per event type
SELECT
    event_type,
    COUNT(*) AS total_events
FROM order_timeline_events
GROUP BY event_type
ORDER BY total_events DESC;


-- Full timeline for a single order (change the ID as needed)
SELECT
    ote.event_timestamp,
    ote.event_type,
    ote.title,
    ote.description,
    ote.executed_by
FROM order_timeline_events ote
WHERE ote.order_id = 1
ORDER BY ote.event_timestamp;


-- Orders with the most timeline events (most activity)
SELECT
    o.order_number,
    o.status,
    COUNT(ote.id) AS event_count
FROM transportation_orders o
JOIN order_timeline_events ote ON ote.order_id = o.id
GROUP BY o.id, o.order_number, o.status
ORDER BY event_count DESC
LIMIT 10;
