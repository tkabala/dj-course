-- Vehicle assignment count and total hours in use
SELECT
    v.id,
    v.make,
    v.model,
    v.year,
    COUNT(a.id)                         AS total_assignments,
    ROUND(SUM(
        EXTRACT(EPOCH FROM upper(a.booking_period) - lower(a.booking_period)) / 3600
    )::numeric, 1)                      AS total_hours_booked
FROM vehicles v
LEFT JOIN assignments a ON a.vehicle_id = v.id
GROUP BY v.id, v.make, v.model, v.year
ORDER BY total_assignments DESC;


-- Vehicles on blackout
SELECT
    v.make || ' ' || v.model || ' (' || v.year || ')'  AS vehicle,
    rb.reason,
    rb.blackout_period
FROM resource_blackouts rb
JOIN vehicles v ON v.id = rb.vehicle_id
WHERE rb.vehicle_id IS NOT NULL
ORDER BY lower(rb.blackout_period);


-- Fleet breakdown by make
SELECT
    make,
    COUNT(*)            AS count,
    MIN(year)           AS oldest,
    MAX(year)           AS newest,
    AVG(fuel_tank_capacity) AS avg_tank_liters
FROM vehicles
GROUP BY make
ORDER BY count DESC;
