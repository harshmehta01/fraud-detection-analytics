-- ============================================================
-- Fintech Transaction Fraud Analysis — SQL Queries (PostgreSQL)
-- Fully annotated version — every section explains what it does
-- and why, directly above the code it refers to.
-- ============================================================


-- ============================================================
-- TABLE SETUP
-- Defines the table structure before any data is imported.
-- Column order matches the exact order columns appear in
-- scored_transactions.csv (output by fraud_detection_model.py),
-- since COPY/\copy match columns by position, not by name.
-- v1-v28 are the anonymized PCA features from the original
-- dataset — not used in any analysis query below, but they must
-- be declared here or the import will fail on column mismatch.
-- ============================================================
CREATE TABLE transactions (
    v1 NUMERIC, v2 NUMERIC, v3 NUMERIC, v4 NUMERIC, v5 NUMERIC,
    v6 NUMERIC, v7 NUMERIC, v8 NUMERIC, v9 NUMERIC, v10 NUMERIC,
    v11 NUMERIC, v12 NUMERIC, v13 NUMERIC, v14 NUMERIC, v15 NUMERIC,
    v16 NUMERIC, v17 NUMERIC, v18 NUMERIC, v19 NUMERIC, v20 NUMERIC,
    v21 NUMERIC, v22 NUMERIC, v23 NUMERIC, v24 NUMERIC, v25 NUMERIC,
    v26 NUMERIC, v27 NUMERIC, v28 NUMERIC,
    amount NUMERIC,                    -- transaction value; NUMERIC keeps exact decimal precision
    hour_of_day INT,                   -- pre-computed hour, exported from Python (0-23)
    transaction_id INT,                -- unique row identifier
    txn_time TIMESTAMP,                -- full date+time, enables EXTRACT()/DATE() functions later
    merchant_category TEXT,            -- synthetic business dimension (see project README)
    location TEXT,                     -- synthetic business dimension (see project README)
    is_fraud BOOLEAN,                  -- true/false ground-truth label
    predicted_fraud_probability NUMERIC, -- model's output score (0-1) from Python
    flagged_as_fraud INT               -- model's final decision (0/1) at the chosen threshold
);


-- ============================================================
-- VERIFICATION QUERIES
-- Run these immediately after importing the CSV, before trusting
-- any analysis query built on top of this table.
-- ============================================================

-- Confirms total row count matches the Python test set size (56,962).
-- If it doesn't, the import didn't fully complete.
SELECT COUNT(*) FROM transactions;

-- GROUP BY is_fraud splits rows into two buckets (true/false) and
-- COUNT(*) counts each bucket. Should return ~98 fraud / ~56,864 legit,
-- matching the Python evaluation output exactly. A mismatch usually
-- means columns shifted during import (wrong column order).
SELECT is_fraud, COUNT(*)
FROM transactions
GROUP BY is_fraud;

-- Visual gut-check on a handful of rows: do amounts look like real
-- currency, is txn_time a valid timestamp, are merchant_category/
-- location populated (not NULL/blank)?
SELECT * FROM transactions LIMIT 5;


-- ============================================================
-- 1. TRANSACTION VOLUME & FRAUD RATE BY HOUR OF DAY
-- Finds WHEN fraud is most likely to occur — useful for staffing
-- a fraud review team or adding a time-based signal to the model.
-- ============================================================
SELECT
    EXTRACT(HOUR FROM txn_time) AS hour_of_day,  -- pulls just the hour (0-23) out of the full timestamp
    COUNT(*) AS total_txns,                       -- total transactions in that hour
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_txns,
        -- CASE WHEN converts each row to 1 (fraud) or 0 (not), since you
        -- can't SUM() a boolean directly — SUM() then counts only fraud rows
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4
        -- 100.0 (not 100) forces decimal division instead of integer division,
        -- which would otherwise round everything down to 0.
        -- Rounded to 4 decimals since fraud rates here are tiny (~0.17%)
    ) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(SUM(CASE WHEN is_fraud THEN amount ELSE 0 END), 2) AS fraud_amount
        -- same conditional-sum trick, but summing € amount instead of
        -- counting rows — gives total € lost to fraud per hour
FROM transactions
GROUP BY EXTRACT(HOUR FROM txn_time)   -- collapses all rows into one row per hour
ORDER BY hour_of_day;


-- ============================================================
-- 2. TRANSACTION VOLUME & FRAUD RATE BY MERCHANT CATEGORY
-- Same pattern as Query 1, grouped by merchant_category instead.
-- CAVEAT: merchant_category was randomly assigned in Python, so
-- any pattern here is coincidental, not a real business signal —
-- still useful to demonstrate the SQL technique and populate a
-- Power BI visual.
-- ============================================================
SELECT
    merchant_category,
    COUNT(*) AS total_txns,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_txns,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4
    ) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_txn_amount,
        -- average transaction size per category — gives context, since
        -- e.g. "Electronics" naturally runs higher than "Groceries"
    ROUND(SUM(CASE WHEN is_fraud THEN amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;   -- highest-risk category first


-- ============================================================
-- 3. TRANSACTION VOLUME & FRAUD RATE BY LOCATION
-- Same pattern again, grouped by location (also synthetic — see
-- caveat in Query 2).
-- ============================================================
SELECT
    location,
    COUNT(*) AS total_txns,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_txns,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4
    ) AS fraud_rate_pct
FROM transactions
GROUP BY location
ORDER BY fraud_rate_pct DESC
LIMIT 15;   -- caps to the 15 riskiest locations, since the full list could get long


-- ============================================================
-- 4. Z-SCORE OUTLIER DETECTION ON TRANSACTION AMOUNT
-- A simple, explainable rule-based fraud signal (no ML needed).
-- Flags any transaction whose amount is more than 3 standard
-- deviations from the mean, calculated separately per merchant
-- category since "normal" spend varies a lot by category.
-- ============================================================
WITH stats AS (
    -- Step 1: calculate each category's average amount and spread
    -- (standard deviation), one row per category.
    SELECT
        merchant_category,
        AVG(amount) AS mean_amount,
        STDDEV(amount) AS stddev_amount
    FROM transactions
    GROUP BY merchant_category
),
scored AS (
    -- Step 2: join each individual transaction to its category's
    -- mean/stddev, then compute the z-score:
    -- z = (value - mean) / standard_deviation
    -- This tells you how many standard deviations a transaction's
    -- amount is from the average for its own category.
    SELECT
        t.transaction_id,
        t.merchant_category,
        t.amount,
        t.is_fraud,
        ROUND(
            (t.amount - s.mean_amount) / NULLIF(s.stddev_amount, 0), 2
            -- NULLIF(x, 0) guards against divide-by-zero: if a category
            -- somehow had zero variance, this turns the 0 denominator
            -- into NULL, so the result is NULL instead of crashing
        ) AS z_score
    FROM transactions t
    JOIN stats s ON t.merchant_category = s.merchant_category
)
-- Step 3: keep only transactions more than 3 standard deviations from
-- their category's mean, in either direction (ABS() covers both
-- unusually high AND unusually low). 3 stddev is a common statistical
-- threshold for "genuine outlier."
SELECT *
FROM scored
WHERE ABS(z_score) > 3
ORDER BY ABS(z_score) DESC;   -- most extreme outliers first


-- ============================================================
-- 5. IQR OUTLIER DETECTION ON TRANSACTION AMOUNT
-- An alternative outlier method to z-score, using quartiles
-- instead of mean/stddev — more robust to extreme skew (fraud
-- amounts are typically heavily skewed, mostly small with a few
-- huge outliers, which can distort the mean/stddev used above).
-- ============================================================
WITH quartiles AS (
    -- Step 1: find each category's Q1 (25th percentile) and Q3
    -- (75th percentile) transaction amount.
    -- PERCENTILE_CONT interpolates between data points for a smooth
    -- result. It's an "ordered-set aggregate" in Postgres, which only
    -- works paired with GROUP BY — NOT with OVER() as a window
    -- function (that combination throws an error in Postgres, even
    -- though some other databases allow it).
    SELECT
        merchant_category,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) AS q3
    FROM transactions
    GROUP BY merchant_category
),
bounds AS (
    -- Step 2: compute the "fences" beyond which a value counts as an
    -- outlier — the classic Tukey's fence rule. 1.5x IQR is a standard
    -- convention (same rule used to draw box-plot whiskers).
    SELECT
        merchant_category,
        q1,
        q3,
        (q3 - q1) AS iqr,                      -- Interquartile Range: spread of the middle 50% of data
        q1 - 1.5 * (q3 - q1) AS lower_fence,
        q3 + 1.5 * (q3 - q1) AS upper_fence
    FROM quartiles
)
-- Step 3: join each transaction to its category's fences, then keep
-- only the ones that fall outside those fences on either side.
SELECT
    t.transaction_id,
    t.merchant_category,
    t.amount,
    t.is_fraud,
    b.lower_fence,
    b.upper_fence
FROM transactions t
JOIN bounds b ON t.merchant_category = b.merchant_category
WHERE t.amount < b.lower_fence OR t.amount > b.upper_fence
ORDER BY t.amount DESC;


-- ============================================================
-- 6. RUNNING FRAUD RATE OVER TIME (for a trend line in Power BI)
-- Smooths daily noise into a rolling trend — the kind of measure
-- a real fraud team would watch to catch a fraud SPIKE early,
-- rather than reacting to one noisy day.
-- KNOWN LIMITATION: this test set spans only a narrow time window
-- from the original 2-day dataset, so you may not get a full 7
-- distinct dates — note this honestly in your README rather than
-- presenting a flat/misleading trend.
-- ============================================================
WITH daily AS (
    -- Step 1: collapse every transaction down to one row per calendar
    -- day. DATE(txn_time) strips the time portion, keeping just the date.
    SELECT
        DATE(txn_time) AS txn_date,
        COUNT(*) AS total_txns,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_txns
    FROM transactions
    GROUP BY DATE(txn_time)
)
-- Step 2: a TRUE window function (unlike the GROUP BY aggregates used
-- everywhere else above). OVER() keeps every row individually, but lets
-- you calculate across a "sliding window" of nearby rows instead of
-- collapsing them. ROWS BETWEEN 6 PRECEDING AND CURRENT ROW = the
-- current day plus the 6 days before it = a 7-day rolling window.
SELECT
    txn_date,
    total_txns,
    fraud_txns,
    ROUND(
        100.0 * SUM(fraud_txns) OVER (
            ORDER BY txn_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )
        / NULLIF(SUM(total_txns) OVER (
            ORDER BY txn_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 0), 4
        -- same NULLIF divide-by-zero guard as Query 4
    ) AS rolling_7day_fraud_rate_pct
FROM daily
ORDER BY txn_date;


-- ============================================================
-- 7. TOP 20 HIGHEST-RISK TRANSACTIONS (for the Power BI "flagged transactions" table)
-- Combines the model's predicted probability (loaded back into SQL after
-- scoring in Python — see fraud_detection_model.py Step 5) with actual outcome.
-- This is the "watchlist" a fraud reviewer would actually look at first.
-- ============================================================
SELECT
    transaction_id,
    txn_time,
    merchant_category,
    location,
    amount,
    predicted_fraud_probability,
    is_fraud AS actual_fraud
        -- renamed in the output only (doesn't change the underlying
        -- table) for clarity, since this query compares the model's
        -- PREDICTED probability against the ACTUAL outcome side by side
FROM transactions
WHERE predicted_fraud_probability IS NOT NULL
    -- filters out any unscored rows — not relevant here since this
    -- table is fully scored, but a safe guard if the table ever mixed
    -- scored and unscored data
ORDER BY predicted_fraud_probability DESC   -- highest-risk transactions first
LIMIT 20;