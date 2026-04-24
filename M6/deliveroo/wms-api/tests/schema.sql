CREATE TABLE customer (
    customer_id   SERIAL PRIMARY KEY,
    name          VARCHAR NOT NULL,
    status        VARCHAR,
    tax_id_number VARCHAR,
    is_deleted    BOOLEAN DEFAULT false,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE customer_contact (
    contact_id  SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customer(customer_id),
    type        VARCHAR NOT NULL,
    details     VARCHAR NOT NULL
);

CREATE TABLE customer_address (
    address_id     SERIAL PRIMARY KEY,
    customer_id    INTEGER REFERENCES customer(customer_id),
    street_address VARCHAR,
    city           VARCHAR,
    country        VARCHAR,
    postal_code    VARCHAR,
    address_type   VARCHAR
);
