-- =====================================================================
-- Read-only role for the AI Copilot's database connection.
--
-- The application-layer SQL safety checks (ai/sql_safety.py) are the
-- primary defense against the copilot running write operations, but
-- defense-in-depth means the database connection itself should also be
-- incapable of writing, in case of an application bug or a
-- not-yet-anticipated bypass. Run this once against your database, then
-- point the copilot's DB connection at this role instead of the main
-- 'supplysense' user (e.g. a separate DB_USER/DB_PASSWORD pair used only
-- by ai/sql_executor.py, configured via its own environment variables).
--
-- Usage:
--   psql -d supplysense -f database/create_readonly_role.sql
-- =====================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'supplysense_readonly') THEN
        CREATE ROLE supplysense_readonly WITH LOGIN PASSWORD 'change_me_in_production';
    END IF;
END
$$;

-- GRANT CONNECT requires the literal database name, but the actual
-- database name varies by environment (e.g. "supplysense" locally,
-- but a provider-assigned name like "neondb" on managed Postgres hosts
-- such as Neon). Using current_database() dynamically here makes this
-- script portable across any environment without editing it.
DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO supplysense_readonly', current_database());
END
$$;
GRANT USAGE ON SCHEMA public TO supplysense_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO supplysense_readonly;

-- Ensure any tables created in the future are also read-only for this role
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO supplysense_readonly;

-- Explicitly confirm no write privileges (belt-and-suspenders; SELECT-only
-- grants above already exclude these, but this documents intent clearly)
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
    ON ALL TABLES IN SCHEMA public FROM supplysense_readonly;
