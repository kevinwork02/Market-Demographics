# Databricks notebook source
# MAGIC %md
# MAGIC # Market Demographics — Development Notebook
# MAGIC
# MAGIC ## Purpose
# MAGIC Prototype queries and validate data before wiring into the Streamlit app at `/Repos/kevin.lynch@locality.com/Market-Demographics`.
# MAGIC
# MAGIC ## Key Tables
# MAGIC
# MAGIC | Table | Key | Role |
# MAGIC |---|---|---|
# MAGIC | `locality_dev.silver.experian_location` | `luid` | Authoritative DMA + zipcode spine — `luid`, `dma`, `zipcode` |
# MAGIC | `locality_dev.gold.experian_marketing_attributes` | `recd_luid` | Age/gender/ethnicity/income/education — 61 cols. Filter `reliability_code BETWEEN 1 AND 4` |
# MAGIC | `locality_dev.default.dma_codes_v3` | `dma_code` | DMA code → name lookup |
# MAGIC
# MAGIC ## Join Spine
# MAGIC `experian_location.luid` → `experian_marketing_attributes.recd_luid` (direct STRING match)
# MAGIC
# MAGIC ## Key Columns
# MAGIC - **Age**: `exact_age` (INT)
# MAGIC - **Gender**: `gender` (STRING: Male, Female, Unknown)
# MAGIC - **Ethnicity**: `ethnic_group` (STRING: Western European, Hispanic, African American, etc.)
# MAGIC - **HHI**: `est_income_amt_thousands` (INT, $6K–$2500K)
# MAGIC - **Education**: `education_level` (STRING: Graduate Degree, Completed College, Some College, High School Diploma, Less Than High School Diploma)
# MAGIC - **Filter**: `reliability_code BETWEEN 1 AND 4`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DMA universe with HH counts
# MAGIC SELECT
# MAGIC     d.dma_code,
# MAGIC     d.dma_name,
# MAGIC     COUNT(DISTINCT el.luid) AS hh_count
# MAGIC FROM locality_dev.silver.experian_location el
# MAGIC JOIN locality_dev.default.dma_codes_v3 d ON el.dma = CAST(d.dma_code AS STRING)
# MAGIC WHERE el.dma IS NOT NULL
# MAGIC GROUP BY d.dma_code, d.dma_name
# MAGIC ORDER BY hh_count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Zip codes within a DMA (prototype for the app's zip selector)
# MAGIC -- Change the DMA code to explore other markets
# MAGIC SELECT
# MAGIC     el.zipcode,
# MAGIC     COUNT(DISTINCT el.luid) AS hh_count
# MAGIC FROM locality_dev.silver.experian_location el
# MAGIC WHERE el.dma = '501'  -- New York
# MAGIC   AND el.zipcode IS NOT NULL
# MAGIC GROUP BY el.zipcode
# MAGIC ORDER BY hh_count DESC
# MAGIC LIMIT 50

# COMMAND ----------

# Market Demographics charts for a selected DMA (+ optional zip filter)
# Prototype for the Streamlit app's main view

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_DMA = "501"   # New York; change as needed
ZIP_FILTER = []       # Empty = all zips in DMA; or e.g. ["10001", "10002"]

NAVY, CYAN, LIGHT_CYAN, LIME, DARK_BG, BORDER = (
    "#1B2A4A", "#00BCD4", "#80DEEA", "#C5E063", "#0d1f3a", "#2a3d5e"
)

zip_clause = ""
if ZIP_FILTER:
    zips = ", ".join(f"'{z}'" for z in ZIP_FILTER)
    zip_clause = f"AND el.zipcode IN ({zips})"

# ── Load demographics ─────────────────────────────────────────────────────────
query = f"""
SELECT
    ma.exact_age,
    ma.gender,
    ma.ethnic_group,
    ma.est_income_amt_thousands,
    ma.education_level
FROM locality_dev.silver.experian_location el
JOIN locality_dev.gold.experian_marketing_attributes ma
    ON ma.recd_luid = el.luid
WHERE el.dma = '{TARGET_DMA}'
  AND ma.reliability_code BETWEEN 1 AND 4
  {zip_clause}
"""

df = spark.sql(query).toPandas()
print(f"DMA {TARGET_DMA} | {len(df):,} persons loaded")

# ── 1. Age Distribution ───────────────────────────────────────────────────────
age_df = df[df["exact_age"].notna()].copy()
bins = [0, 18, 25, 35, 45, 55, 65, 75, 120]
labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
age_df["age_band"] = pd.cut(age_df["exact_age"], bins=bins, labels=labels, right=False)
age_counts = age_df["age_band"].value_counts().sort_index()

fig_age = go.Figure(go.Bar(
    x=age_counts.index.astype(str).tolist(),
    y=age_counts.values.tolist(),
    marker_color=CYAN,
    text=[f"{v:,.0f}" for v in age_counts.values],
    textposition="outside"
))
fig_age.update_layout(
    title=dict(text="Age Distribution", font_color=LIGHT_CYAN),
    xaxis=dict(title="Age Band", gridcolor=BORDER, title_font_color=LIGHT_CYAN),
    yaxis=dict(title="Persons", gridcolor=BORDER, title_font_color=LIGHT_CYAN),
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color=LIGHT_CYAN,
    height=380, margin=dict(t=55, b=40)
)
fig_age.show()

# ── 2. Gender Distribution ────────────────────────────────────────────────────
gender_df = df[df["gender"].notna() & (df["gender"] != "Unknown")]
gender_counts = gender_df["gender"].value_counts()

fig_gender = go.Figure(go.Pie(
    labels=gender_counts.index.tolist(),
    values=gender_counts.values.tolist(),
    marker_colors=[CYAN, LIME],
    hole=0.4,
    textinfo="label+percent",
    textfont_color=LIGHT_CYAN
))
fig_gender.update_layout(
    title=dict(text="Gender Split", font_color=LIGHT_CYAN),
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color=LIGHT_CYAN,
    height=380, margin=dict(t=55, b=40)
)
fig_gender.show()

# ── 3. Ethnicity Distribution ─────────────────────────────────────────────────
eth_df = df[df["ethnic_group"].notna() & (df["ethnic_group"] != "Uncoded")]
eth_counts = eth_df["ethnic_group"].value_counts().head(10)

fig_eth = go.Figure(go.Bar(
    x=eth_counts.values.tolist(),
    y=eth_counts.index.tolist(),
    orientation="h",
    marker_color=CYAN,
    text=[f"{v:,.0f}" for v in eth_counts.values],
    textposition="outside"
))
fig_eth.update_layout(
    title=dict(text="Ethnicity (Top 10)", font_color=LIGHT_CYAN),
    xaxis=dict(title="Persons", gridcolor=BORDER, title_font_color=LIGHT_CYAN),
    yaxis=dict(gridcolor=BORDER),
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color=LIGHT_CYAN,
    height=420, margin=dict(l=180, t=55, b=40)
)
fig_eth.show()

# ── 4. Household Income Distribution ─────────────────────────────────────────
inc_df = df[df["est_income_amt_thousands"].notna()].copy()
inc_bins = [0, 25, 50, 75, 100, 150, 200, 300, 5000]
inc_labels = ["<$25K", "$25-50K", "$50-75K", "$75-100K", "$100-150K", "$150-200K", "$200-300K", "$300K+"]
inc_df["income_band"] = pd.cut(inc_df["est_income_amt_thousands"], bins=inc_bins, labels=inc_labels, right=False)
inc_counts = inc_df["income_band"].value_counts().sort_index()

fig_inc = go.Figure(go.Bar(
    x=inc_counts.index.astype(str).tolist(),
    y=inc_counts.values.tolist(),
    marker_color=LIME,
    text=[f"{v:,.0f}" for v in inc_counts.values],
    textposition="outside"
))
fig_inc.update_layout(
    title=dict(text="Household Income Distribution", font_color=LIGHT_CYAN),
    xaxis=dict(title="Income Band", gridcolor=BORDER, title_font_color=LIGHT_CYAN, tickangle=-30),
    yaxis=dict(title="Persons", gridcolor=BORDER, title_font_color=LIGHT_CYAN),
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color=LIGHT_CYAN,
    height=400, margin=dict(t=55, b=80)
)
fig_inc.show()

# ── 5. Education Distribution ─────────────────────────────────────────────────
edu_df = df[df["education_level"].notna()]
edu_order = ["Less Than High School Diploma", "High School Diploma", "Some College", "Completed College", "Graduate Degree"]
edu_counts = edu_df["education_level"].value_counts().reindex(edu_order).dropna()

fig_edu = go.Figure(go.Bar(
    x=edu_counts.index.tolist(),
    y=edu_counts.values.tolist(),
    marker_color=CYAN,
    text=[f"{v:,.0f}" for v in edu_counts.values],
    textposition="outside"
))
fig_edu.update_layout(
    title=dict(text="Education Level", font_color=LIGHT_CYAN),
    xaxis=dict(title="Education", gridcolor=BORDER, title_font_color=LIGHT_CYAN, tickangle=-20),
    yaxis=dict(title="Persons", gridcolor=BORDER, title_font_color=LIGHT_CYAN),
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color=LIGHT_CYAN,
    height=400, margin=dict(t=55, b=80)
)
fig_edu.show()

print(f"\n--- Summary ---")
print(f"Total persons: {len(df):,}")
print(f"Age coverage: {age_df.shape[0]:,} ({age_df.shape[0]/len(df)*100:.1f}%)")
print(f"Income coverage: {inc_df.shape[0]:,} ({inc_df.shape[0]/len(df)*100:.1f}%)")
print(f"Education coverage: {edu_df.shape[0]:,} ({edu_df.shape[0]/len(df)*100:.1f}%)")
