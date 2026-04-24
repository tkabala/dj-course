import os
import sys
import pytest
import psycopg2

SCHEMA_SQL = os.path.join(os.path.dirname(__file__), 'schema.sql')


@pytest.fixture(scope="session")
def postgres_container():
    from testcontainers.postgres import PostgresContainer
    with PostgresContainer("postgres:15-alpine") as pg:
        dsn = pg.get_connection_url().replace("+psycopg2", "")
        os.environ['POSTGRES_URL'] = dsn
        os.environ['SERVICE_NAME'] = 'wms-api-test'
        yield pg


@pytest.fixture(scope="session")
def flask_app(postgres_container):
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from application import app
    app.config['TESTING'] = True
    yield app


@pytest.fixture(scope="session")
def db_schema(postgres_container):
    dsn = os.environ['POSTGRES_URL']
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with open(SCHEMA_SQL) as f:
        conn.cursor().execute(f.read())
    conn.close()


@pytest.fixture
def client(flask_app, db_schema):
    dsn = os.environ['POSTGRES_URL']
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO customer (customer_id, name, status, tax_id_number, is_deleted)
        VALUES
            (1, 'Alpha Corp', 'active',   '111-11-1111', false),
            (2, 'Beta LLC',   'inactive', '222-22-2222', false)
        ON CONFLICT (customer_id) DO UPDATE
            SET name=EXCLUDED.name, status=EXCLUDED.status,
                is_deleted=false, updated_at=CURRENT_TIMESTAMP;
    """)
    yield flask_app.test_client()
    cur.execute("DELETE FROM customer_contact;")
    cur.execute("DELETE FROM customer_address;")
    cur.execute("TRUNCATE customer RESTART IDENTITY CASCADE;")
    cur.close()
    conn.close()
