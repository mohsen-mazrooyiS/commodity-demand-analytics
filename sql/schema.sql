-- ============================================================================
-- Armani Middle East Trading — Star Schema
-- Fact table: fact_sales (one row per product-week)
-- Dimensions: dim_product, dim_date
--
-- NOTE: the real Sample_Data.xlsx has no Region, Customer_Segment, or
-- customer/credit-level fields, so those dimensions from the v1 draft
-- schema are dropped. See README "Data limitations" section for how this
-- affects the Sales & Credit KPIs (DSO, Customer Default Risk).
-- ============================================================================

DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_product;
DROP TABLE IF EXISTS dim_date;

CREATE TABLE dim_product (
    product_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_date (
    date_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    calendar_date DATE UNIQUE NOT NULL,
    week_number   INTEGER UNIQUE NOT NULL,
    month         INTEGER NOT NULL,
    quarter       INTEGER NOT NULL,
    year          INTEGER NOT NULL
);

CREATE TABLE fact_sales (
    sale_id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    date_id                   INTEGER NOT NULL REFERENCES dim_date(date_id),
    product_id                INTEGER NOT NULL REFERENCES dim_product(product_id),

    sales_units               REAL NOT NULL,   -- inventory-censored: see is_stockout
    cogs_per_unit             REAL NOT NULL,
    price_per_unit            REAL NOT NULL,
    inventory_level           REAL NOT NULL,
    demand_actual             REAL NOT NULL,   -- uncensored demand estimate - the forecasting target

    revenue                   REAL NOT NULL,
    gross_profit              REAL NOT NULL,
    operating_profit          REAL NOT NULL,
    inventory_turnover_ratio  REAL,
    gross_profit_margin       REAL,
    revenue_growth            REAL,            -- capped, see cleaning report for winsorization details

    is_stockout               INTEGER NOT NULL DEFAULT 0,  -- 1 if Sales_Units=0 & Inventory_Level=0

    UNIQUE(date_id, product_id)
);

CREATE INDEX idx_fact_sales_product ON fact_sales(product_id);
CREATE INDEX idx_fact_sales_date ON fact_sales(date_id);
