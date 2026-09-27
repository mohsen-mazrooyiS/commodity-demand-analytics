-- ============================================================================
-- Armani Middle East Trading — Star Schema
-- Fact table: fact_sales (one row per product-week)
-- Dimensions: dim_product, dim_date, dim_customer_segment, dim_region
-- ============================================================================

DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_product;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_customer_segment;
DROP TABLE IF EXISTS dim_region;

CREATE TABLE dim_product (
    product_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_region (
    region_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    region_name TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_customer_segment (
    segment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_name TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_date (
    date_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    calendar_date DATE UNIQUE NOT NULL,
    week_number   INTEGER NOT NULL,
    month         INTEGER NOT NULL,
    quarter       INTEGER NOT NULL,
    year          INTEGER NOT NULL
);

CREATE TABLE fact_sales (
    sale_id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    date_id                     INTEGER NOT NULL REFERENCES dim_date(date_id),
    product_id                  INTEGER NOT NULL REFERENCES dim_product(product_id),
    region_id                   INTEGER NOT NULL REFERENCES dim_region(region_id),
    segment_id                  INTEGER NOT NULL REFERENCES dim_customer_segment(segment_id),

    sales_units                 REAL NOT NULL,
    cogs_per_unit                REAL NOT NULL,
    price_per_unit              REAL NOT NULL,
    inventory_level             REAL NOT NULL,
    days_sales_outstanding_input REAL,
    customer_credit_limit       REAL,
    customer_outstanding_balance REAL,
    demand_actual               REAL NOT NULL,

    -- derived financial fields, computed once during load (not re-derived
    -- ad-hoc downstream, so every consumer sees the same numbers)
    revenue                     REAL NOT NULL,
    gross_profit                REAL NOT NULL,

    UNIQUE(date_id, product_id, region_id, segment_id)
);

CREATE INDEX idx_fact_sales_product ON fact_sales(product_id);
CREATE INDEX idx_fact_sales_date ON fact_sales(date_id);
