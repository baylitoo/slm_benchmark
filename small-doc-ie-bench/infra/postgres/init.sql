-- Extensions kept minimal for portability.
CREATE TABLE IF NOT EXISTS service_bootstrap_marker (
    id integer PRIMARY KEY,
    created_at timestamptz DEFAULT now()
);
INSERT INTO service_bootstrap_marker (id) VALUES (1) ON CONFLICT DO NOTHING;

-- Dedicated database for the self-hosted Inngest server (shares this Postgres
-- instance instead of standing up a second one). `\gexec` makes it idempotent;
-- the connecting role (docie) owns it, matching INNGEST_POSTGRES_URI.
SELECT 'CREATE DATABASE inngest'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'inngest')\gexec
