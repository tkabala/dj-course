-- ─────────────────────────────────────────────
-- Resource Availability Report
-- Default: tomorrow 10:00–12:00
-- Override: task query PERIOD="[2026-02-21 08:00,2026-02-21 20:00)" -- resource-availability-report
-- ─────────────────────────────────────────────
\if :{?period}
\else
SELECT '[' || ((current_date + 1)::timestamp + time '10:00')::text
          || ',' || ((current_date + 1)::timestamp + time '12:00')::text
          || ')' AS period \gset
\endif

\set QUIET on
SET client_min_messages = WARNING;

\echo ''
\echo 'Period:' :period

-- Build blocking data once into temp tables
DROP TABLE IF EXISTS _driver_blocks;
DROP TABLE IF EXISTS _vehicle_blocks;

CREATE TEMP TABLE _driver_blocks AS
    SELECT a.driver_id, 'Assigned to order ' || o.order_number AS reason
    FROM assignments a
    JOIN transportation_orders o ON o.id = a.order_id
    WHERE a.booking_period && :'period'::tsrange
    UNION ALL
    SELECT rb.driver_id,
           'Blackout: ' || rb.reason
               || ' (' || lower(rb.blackout_period)::date
               || ' – ' || upper(rb.blackout_period)::date || ')'
    FROM resource_blackouts rb
    WHERE rb.driver_id IS NOT NULL
      AND rb.blackout_period && :'period'::tsrange
    UNION ALL
    SELECT d.id, 'Not on shift'
    FROM drivers d
    WHERE NOT EXISTS (
        SELECT 1 FROM driver_shifts ds
        WHERE ds.driver_id = d.id
          AND ds.day_of_week = EXTRACT(DOW FROM lower(:'period'::tsrange))::int
          AND ds.start_time <= lower(:'period'::tsrange)::time
          AND ds.end_time   >= upper(:'period'::tsrange)::time
    )
    AND d.id NOT IN (
        SELECT a.driver_id FROM assignments a
        WHERE a.booking_period && :'period'::tsrange
        UNION
        SELECT rb.driver_id FROM resource_blackouts rb
        WHERE rb.driver_id IS NOT NULL
          AND rb.blackout_period && :'period'::tsrange
    );

CREATE TEMP TABLE _vehicle_blocks AS
    SELECT a.vehicle_id, 'Assigned to order ' || o.order_number AS reason
    FROM assignments a
    JOIN transportation_orders o ON o.id = a.order_id
    WHERE a.booking_period && :'period'::tsrange
    UNION ALL
    SELECT rb.vehicle_id,
           'Blackout: ' || rb.reason
               || ' (' || lower(rb.blackout_period)::date
               || ' – ' || upper(rb.blackout_period)::date || ')'
    FROM resource_blackouts rb
    WHERE rb.vehicle_id IS NOT NULL
      AND rb.blackout_period && :'period'::tsrange;


\echo ''
\pset title 'AVAILABLE DRIVERS'
SELECT
    d.first_name || ' ' || d.last_name       AS driver,
    d.contract_type,
    TO_CHAR(lower(:'period'::tsrange), 'FMDay') || ' ' || ds.start_time || ' – ' || ds.end_time AS shift
FROM drivers d
JOIN driver_shifts ds ON ds.driver_id = d.id
    AND ds.day_of_week = EXTRACT(DOW FROM lower(:'period'::tsrange))::int
    AND ds.start_time <= lower(:'period'::tsrange)::time
    AND ds.end_time   >= upper(:'period'::tsrange)::time
WHERE d.id NOT IN (SELECT driver_id FROM _driver_blocks)
ORDER BY d.last_name;


\pset title 'AVAILABLE VEHICLES'
SELECT
    v.make,
    v.model,
    v.year,
    v.fuel_tank_capacity AS tank_l
FROM vehicles v
WHERE v.id NOT IN (SELECT vehicle_id FROM _vehicle_blocks)
ORDER BY v.make, v.model;


\pset title 'UNAVAILABLE DRIVERS'
SELECT
    d.first_name || ' ' || d.last_name AS driver,
    d.contract_type,
    string_agg(DISTINCT b.reason, ' | ') AS reason
FROM _driver_blocks b
JOIN drivers d ON d.id = b.driver_id
GROUP BY d.id, d.first_name, d.last_name, d.contract_type
ORDER BY d.id;


\pset title 'UNAVAILABLE VEHICLES'
SELECT
    v.make || ' ' || v.model AS vehicle,
    v.year,
    string_agg(DISTINCT b.reason, ' | ') AS reason
FROM _vehicle_blocks b
JOIN vehicles v ON v.id = b.vehicle_id
GROUP BY v.id, v.make, v.model, v.year
ORDER BY v.id;
