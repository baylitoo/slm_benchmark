-- Extensions kept minimal for portability.
CREATE TABLE IF NOT EXISTS service_bootstrap_marker (
    id integer PRIMARY KEY,
    created_at timestamptz DEFAULT now()
);
INSERT INTO service_bootstrap_marker (id) VALUES (1) ON CONFLICT DO NOTHING;
