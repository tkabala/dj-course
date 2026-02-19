-- Drivers with their assignment count and total booking hours
SELECT
    d.id,
    d.first_name || ' ' || d.last_name  AS driver_name,
    d.contract_type,
    COUNT(a.id)                         AS total_assignments,
    ROUND(SUM(
        EXTRACT(EPOCH FROM upper(a.booking_period) - lower(a.booking_period)) / 3600
    )::numeric, 1)                      AS total_hours_booked
FROM drivers d
LEFT JOIN assignments a ON a.driver_id = d.id
GROUP BY d.id, d.first_name, d.last_name, d.contract_type
ORDER BY total_assignments DESC;


-- Driver shift schedule (days 0=Sun .. 6=Sat)
SELECT
    d.first_name || ' ' || d.last_name  AS driver_name,
    CASE ds.day_of_week
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END                                 AS day,
    ds.start_time,
    ds.end_time
FROM driver_shifts ds
JOIN drivers d ON d.id = ds.driver_id
ORDER BY d.last_name, ds.day_of_week;


-- Drivers currently on blackout
SELECT
    d.first_name || ' ' || d.last_name  AS driver_name,
    rb.reason,
    rb.blackout_period
FROM resource_blackouts rb
JOIN drivers d ON d.id = rb.driver_id
WHERE rb.driver_id IS NOT NULL
ORDER BY lower(rb.blackout_period);
