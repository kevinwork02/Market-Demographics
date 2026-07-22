"""Market Demographics — Streamlit App
DMA + zip-level demographic profiling from Experian Marketing Attributes.
"""
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from databricks import sql as dbsql
from dotenv import load_dotenv

load_dotenv()

# ── Theme ─────────────────────────────────────────────────────────────────────
NAVY = "#1B2A4A"
CYAN = "#00BCD4"
LIGHT_CYAN = "#80DEEA"
LIME = "#C5E063"
DARK_BG = "#0d1f3a"
BORDER = "#2a3d5e"

# ── Credentials ───────────────────────────────────────────────────────────────
def _get_creds():
    """Try st.secrets first, then env vars."""
    try:
        host = st.secrets["DATABRICKS_SERVER_HOSTNAME"]
        path = st.secrets["DATABRICKS_HTTP_PATH"]
        token = st.secrets["DATABRICKS_ACCESS_TOKEN"]
    except (KeyError, FileNotFoundError):
        host = os.environ["DATABRICKS_SERVER_HOSTNAME"]
        path = os.environ["DATABRICKS_HTTP_PATH"]
        token = os.environ["DATABRICKS_ACCESS_TOKEN"]
    return host, path, token


def _run_query(query: str) -> pd.DataFrame:
    host, path, token = _get_creds()
    with dbsql.connect(server_hostname=host, http_path=path, access_token=token) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


# ── Cached Loaders ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_dma_list() -> pd.DataFrame:
    return _run_query("""
        SELECT d.dma_code, d.dma_name, COUNT(DISTINCT el.luid) AS hh_count
        FROM locality_dev.silver.experian_location el
        JOIN locality_dev.default.dma_codes_v3 d ON el.dma = CAST(d.dma_code AS STRING)
        WHERE el.dma IS NOT NULL
        GROUP BY d.dma_code, d.dma_name
        ORDER BY hh_count DESC
    """)


@st.cache_data(ttl=3600)
def load_zip_codes(dma_codes: tuple) -> pd.DataFrame:
    dma_filter = ", ".join(f"'{c}'" for c in dma_codes)
    return _run_query(f"""
        SELECT el.zipcode, el.dma AS dma_code, COUNT(DISTINCT el.luid) AS hh_count
        FROM locality_dev.silver.experian_location el
        WHERE el.dma IN ({dma_filter})
          AND el.zipcode IS NOT NULL
        GROUP BY el.zipcode, el.dma
        ORDER BY hh_count DESC
    """)


@st.cache_data(ttl=600)
def load_demographics(dma_codes: tuple, zip_codes: tuple) -> pd.DataFrame:
    dma_filter = ", ".join(f"'{c}'" for c in dma_codes)
    zip_clause = ""
    if zip_codes:
        zips = ", ".join(f"'{z}'" for z in zip_codes)
        zip_clause = f"AND el.zipcode IN ({zips})"

    return _run_query(f"""
        SELECT
            ma.exact_age,
            ma.gender,
            ma.ethnic_group,
            ma.est_income_amt_thousands,
            ma.education_level
        FROM locality_dev.silver.experian_location el
        JOIN locality_dev.gold.experian_marketing_attributes ma
            ON ma.recd_luid = el.luid
        WHERE el.dma IN ({dma_filter})
          AND ma.reliability_code BETWEEN 1 AND 4
          {zip_clause}
    """)


# ── Chart Builders ────────────────────────────────────────────────────────────
def _layout(title: str, height: int = 380, **kwargs) -> dict:
    base = dict(
        title=dict(text=title, font_color=LIGHT_CYAN),
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        font_color=LIGHT_CYAN, height=height,
        margin=dict(t=55, b=40)
    )
    base.update(kwargs)
    return base


def chart_age(df: pd.DataFrame) -> go.Figure:
    age_df = df[df["exact_age"].notna()].copy()
    bins = [0, 18, 25, 35, 45, 55, 65, 75, 120]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    age_df["age_band"] = pd.cut(age_df["exact_age"].astype(float), bins=bins, labels=labels, right=False)
    counts = age_df["age_band"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str).tolist(), y=counts.values.tolist(),
        marker_color=CYAN, text=[f"{v:,.0f}" for v in counts.values], textposition="outside"
    ))
    fig.update_layout(**_layout(
        "Age Distribution",
        xaxis=dict(title="Age Band", gridcolor=BORDER, title_font_color=LIGHT_CYAN),
        yaxis=dict(title="Persons", gridcolor=BORDER, title_font_color=LIGHT_CYAN)
    ))
    return fig


def chart_gender(df: pd.DataFrame) -> go.Figure:
    g = df[df["gender"].notna() & (df["gender"] != "Unknown")]
    counts = g["gender"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(),
        marker_colors=[CYAN, LIME], hole=0.4,
        textinfo="label+percent", textfont_color=LIGHT_CYAN
    ))
    fig.update_layout(**_layout("Gender Split"))
    return fig


def chart_ethnicity(df: pd.DataFrame) -> go.Figure:
    eth = df[df["ethnic_group"].notna() & (df["ethnic_group"] != "Uncoded")]
    counts = eth["ethnic_group"].value_counts().head(10)
    fig = go.Figure(go.Bar(
        x=counts.values.tolist(), y=counts.index.tolist(),
        orientation="h", marker_color=CYAN,
        text=[f"{v:,.0f}" for v in counts.values], textposition="outside"
    ))
    fig.update_layout(**_layout(
        "Ethnicity (Top 10)", height=420,
        xaxis=dict(title="Persons", gridcolor=BORDER, title_font_color=LIGHT_CYAN),
        yaxis=dict(gridcolor=BORDER),
        margin=dict(l=180, t=55, b=40)
    ))
    return fig


def chart_income(df: pd.DataFrame) -> go.Figure:
    inc = df[df["est_income_amt_thousands"].notna()].copy()
    bins = [0, 25, 50, 75, 100, 150, 200, 300, 5000]
    labels = ["<$25K", "$25-50K", "$50-75K", "$75-100K", "$100-150K", "$150-200K", "$200-300K", "$300K+"]
    inc["band"] = pd.cut(inc["est_income_amt_thousands"].astype(float), bins=bins, labels=labels, right=False)
    counts = inc["band"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str).tolist(), y=counts.values.tolist(),
        marker_color=LIME, text=[f"{v:,.0f}" for v in counts.values], textposition="outside"
    ))
    fig.update_layout(**_layout(
        "Household Income Distribution", height=400,
        xaxis=dict(title="Income Band", gridcolor=BORDER, title_font_color=LIGHT_CYAN, tickangle=-30),
        yaxis=dict(title="Persons", gridcolor=BORDER, title_font_color=LIGHT_CYAN),
        margin=dict(t=55, b=80)
    ))
    return fig


def chart_education(df: pd.DataFrame) -> go.Figure:
    edu = df[df["education_level"].notna()]
    order = ["Less Than High School Diploma", "High School Diploma", "Some College", "Completed College", "Graduate Degree"]
    counts = edu["education_level"].value_counts().reindex(order).dropna()
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(), y=counts.values.tolist(),
        marker_color=CYAN, text=[f"{v:,.0f}" for v in counts.values], textposition="outside"
    ))
    fig.update_layout(**_layout(
        "Education Level", height=400,
        xaxis=dict(title="Education", gridcolor=BORDER, title_font_color=LIGHT_CYAN, tickangle=-20),
        yaxis=dict(title="Persons", gridcolor=BORDER, title_font_color=LIGHT_CYAN),
        margin=dict(t=55, b=80)
    ))
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Market Demographics", layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {NAVY}; }}
        h1, h2, h3 {{ color: {LIGHT_CYAN}; }}
        .stMetric label {{ color: {LIGHT_CYAN}; }}
        .stMetric [data-testid="stMetricValue"] {{ color: {CYAN}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Market Demographics")
    st.caption("Experian Marketing Attributes — DMA & Zip-level demographic profiling")

    # ── Step 1: DMA Selection ──
    with st.spinner("Loading DMA list..."):
        dma_df = load_dma_list()

    dma_options = {f"{r['dma_name']} ({r['dma_code']})": str(r["dma_code"]) for _, r in dma_df.iterrows()}
    selected_labels = st.multiselect(
        "Select DMAs",
        options=list(dma_options.keys()),
        default=[list(dma_options.keys())[0]],  # default to largest DMA
        help="Choose one or more DMAs to profile"
    )
    selected_dma_codes = tuple(dma_options[lbl] for lbl in selected_labels)

    if not selected_dma_codes:
        st.warning("Please select at least one DMA.")
        return

    # ── Step 2: Optional Zip Code Filter ──
    with st.expander("\U0001F4CD Filter by Zip Code (optional)", expanded=False):
        zip_df = load_zip_codes(selected_dma_codes)
        zip_options = sorted(zip_df["zipcode"].unique().tolist())
        selected_zips = st.multiselect(
            "Select zip codes (leave empty for entire DMA)",
            options=zip_options,
            default=[],
            help=f"{len(zip_options)} zip codes available in selected DMA(s)"
        )
    selected_zip_tuple = tuple(selected_zips) if selected_zips else ()

    # ── Step 3: Load & Chart ──
    with st.spinner("Loading demographics..."):
        df = load_demographics(selected_dma_codes, selected_zip_tuple)

    if df.empty:
        st.error("No data returned for the selected filters.")
        return

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Persons", f"{len(df):,}")
    age_known = df["exact_age"].notna().sum()
    col2.metric("Median Age", f"{int(df['exact_age'].median())}" if age_known > 0 else "N/A")
    inc_known = df["est_income_amt_thousands"].notna().sum()
    median_inc = int(df["est_income_amt_thousands"].median()) if inc_known > 0 else None
    col3.metric("Median HHI", f"${median_inc}K" if median_inc else "N/A")
    col4.metric("DMAs Selected", str(len(selected_dma_codes)))

    # Charts in 2-column layout
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart_age(df), use_container_width=True)
        st.plotly_chart(chart_ethnicity(df), use_container_width=True)
        st.plotly_chart(chart_education(df), use_container_width=True)
    with right:
        st.plotly_chart(chart_gender(df), use_container_width=True)
        st.plotly_chart(chart_income(df), use_container_width=True)


if __name__ == "__main__":
    main()
