-- Business KPI queries on table `policyholders` (SQLite).
-- Each query is tagged with `-- name: <id>` so it can be executed from Python.

-- name: portfolio_overview
SELECT COUNT(*)                         AS policyholders,
       ROUND(SUM(charges), 0)           AS total_charges,
       ROUND(AVG(charges), 0)           AS avg_charges,
       ROUND(AVG(smoker = 'yes') * 100, 1) AS smoker_rate_pct,
       ROUND(AVG(bmi >= 30) * 100, 1)      AS obesity_rate_pct
FROM policyholders;

-- name: cost_by_smoker
SELECT smoker,
       COUNT(*)                                                  AS customers,
       ROUND(AVG(charges), 0)                                    AS avg_charges,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM policyholders), 1)          AS customer_share_pct,
       ROUND(100.0 * SUM(charges) / (SELECT SUM(charges) FROM policyholders), 1)  AS cost_share_pct
FROM policyholders
GROUP BY smoker
ORDER BY avg_charges DESC;

-- name: cost_by_risk_tier
SELECT risk_tier,
       COUNT(*)                AS customers,
       ROUND(AVG(charges), 0)  AS avg_charges,
       ROUND(100.0 * SUM(charges) / (SELECT SUM(charges) FROM policyholders), 1) AS cost_share_pct
FROM policyholders
GROUP BY risk_tier
ORDER BY CASE risk_tier WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 WHEN 'High' THEN 3 ELSE 4 END;

-- name: region_smoker_matrix
SELECT region,
       ROUND(AVG(CASE WHEN smoker = 'yes' THEN charges END), 0) AS avg_smoker,
       ROUND(AVG(CASE WHEN smoker = 'no'  THEN charges END), 0) AS avg_non_smoker,
       ROUND(AVG(smoker = 'yes') * 100, 1)                      AS smoker_rate_pct,
       COUNT(*)                                                 AS customers
FROM policyholders
GROUP BY region
ORDER BY avg_smoker DESC;

-- name: age_group_trend
SELECT age_group,
       COUNT(*)               AS customers,
       ROUND(AVG(charges), 0) AS avg_charges,
       ROUND(AVG(CASE WHEN smoker = 'no' THEN charges END), 0) AS avg_non_smoker
FROM policyholders
GROUP BY age_group
ORDER BY age_group;

-- name: bmi_smoker_interaction
SELECT bmi_category,
       smoker,
       COUNT(*)               AS customers,
       ROUND(AVG(charges), 0) AS avg_charges
FROM policyholders
GROUP BY bmi_category, smoker
ORDER BY smoker DESC, avg_charges DESC;

-- name: top_decile_concentration
-- Pareto check: what share of total cost comes from the top 10% most expensive customers?
WITH ranked AS (
    SELECT charges, NTILE(10) OVER (ORDER BY charges DESC) AS decile
    FROM policyholders
)
SELECT decile,
       COUNT(*)                AS customers,
       ROUND(AVG(charges), 0)  AS avg_charges,
       ROUND(100.0 * SUM(charges) / (SELECT SUM(charges) FROM policyholders), 1) AS cost_share_pct,
       ROUND(SUM(100.0 * SUM(charges) / (SELECT SUM(charges) FROM policyholders))
             OVER (ORDER BY decile), 1) AS cumulative_cost_share_pct
FROM ranked
GROUP BY decile
ORDER BY decile;

-- name: family_size_cost
SELECT children,
       COUNT(*)               AS customers,
       ROUND(AVG(charges), 0) AS avg_charges,
       ROUND(AVG(charges) / NULLIF(children + 1, 0), 0) AS avg_charges_per_covered_person
FROM policyholders
GROUP BY children
ORDER BY children;
