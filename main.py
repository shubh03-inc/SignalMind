
from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st

from src.intelligence.sales_intelligence import build_sales_intelligence
from src.intelligence.revenue_intelligence import build_revenue_intelligence
from src.intelligence.product_performance import build_product_performance
from src.intelligence.business_performance import build_business_performance
from src.gita.predict import predict_gita
from src.gita.upload_intelligence import render_uploaded_gita_tab
from src.upload_validation import (
    read_uploaded_dataset,
    validate_dataset,
    profile_dataset,
)
from src.analysis.upload_sales import analyze_uploaded_sales


# =========================================================
# 1. CONFIGURATION
# =========================================================

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
ASSETS_DIR = PROJECT / "assets"
LOGO_PATH = ASSETS_DIR / "SignalMind_logo.png"
MODEL_PATH = PROJECT / "models" / "gita_isolation_forest.joblib"

REQUIRED_DATA_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
]

st.set_page_config(
    page_title="SignalMind | Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep the file size limit consistent with the uploader.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# =========================================================
# 2. DASHBOARD STYLE
# =========================================================

st.session_state.setdefault("sm_theme", "Light")

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #101827;
    }

    [data-testid="stSidebar"] * {
        color: #e5eaf3;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c3ccdc;
    }

    .sm-hero {
        padding: 28px 30px;
        border-radius: 20px;
        margin-bottom: 22px;
        background: linear-gradient(120deg, #14213d, #244c83);
        color: white;
        border: 1px solid #2c4e7c;
    }

    .sm-eyebrow {
        color: #9fc4ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .sm-hero h1 {
        color: white;
        margin: 8px 0 6px 0;
        font-size: 36px;
        font-weight: 750;
    }

    .sm-hero p {
        color: #d6e2f3;
        margin: 0;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 17px 18px;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #53627a;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 7px;
        border-bottom: 1px solid #dce3ee;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 12px 15px;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }

    .stButton button,
    .stDownloadButton button {
        border-radius: 9px;
        font-weight: 600;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }

    h1, h2, h3 {
        color: #17243a;
    }

    /* Responsive layout: protect the dashboard on laptop and narrow screens. */
    .block-container {
        width: 100%;
        max-width: 1600px;
        padding-top: 3rem;
        padding-bottom: 2rem;
        padding-left: clamp(0.8rem, 2.4vw, 2.2rem);
        padding-right: clamp(0.8rem, 2.4vw, 2.2rem);
    }

    /* Let Streamlit manage column wrapping; global flex-wrap breaks alignment. */
    [data-testid="stHorizontalBlock"] {
        min-width: 0;
    }

    [data-testid="stMetric"] {
        min-width: 0;
        height: 100%;
        overflow-wrap: anywhere;
    }

    [data-testid="stMetricValue"] {
        font-size: clamp(1.1rem, 2vw, 1.8rem);
        overflow-wrap: anywhere;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex;
        max-width: 100%;
        overflow-x: auto;
        overflow-y: hidden;
        flex-wrap: nowrap;
        gap: 0.35rem;
        scrollbar-width: thin;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        white-space: nowrap;
        flex-shrink: 0;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        max-width: 100%;
    }

    .sm-brand-row {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }

    .sm-hero {
        padding: clamp(1.1rem, 3vw, 2rem);
    }

    .sm-hero h1 {
        font-size: clamp(1.8rem, 4vw, 2.6rem);
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.65rem 0.8rem;
            font-size: 0.875rem;
        }
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 0.8rem;
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }
        .sm-hero {
            border-radius: 14px;
            margin-bottom: 1rem;
        }
        [data-testid="stMetric"] {
            padding: 0.8rem;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            padding: 0.55rem 0.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if st.session_state.get("sm_theme", "Light") == "Dark":
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background-color: #0b1220 !important;
            color: #e5eaf3 !important;
        }
        [data-testid="stHeader"] {
            background: rgba(11, 18, 32, 0.96) !important;
        }
        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] li,
        [data-testid="stMain"] span,
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
            color: #e5eaf3;
        }
        div[data-testid="stMetric"],
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stExpander"] {
            background-color: #131e30 !important;
            color: #e5eaf3 !important;
            border-color: #2b3a50 !important;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricValue"] {
            color: #e5eaf3 !important;
        }
        [data-baseweb="input"],
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"],
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploader"] section > div {
            background-color: #131e30 !important;
            color: #e5eaf3 !important;
            border-color: #40516b !important;
        }
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] input {
            color: #e5eaf3 !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            border-bottom-color: #2b3a50 !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            color: #b9c7da !important;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            color: #8dbbff !important;
            border-bottom-color: #8dbbff !important;
        }
        .sm-hero {
            background: linear-gradient(120deg, #101a2d, #173b68) !important;
            border-color: #2c4e7c !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 3. CACHED OLIST PIPELINES
# =========================================================

@st.cache_data(show_spinner=False)
def load_sales():
    return build_sales_intelligence()


@st.cache_data(show_spinner=False)
def load_revenue():
    return build_revenue_intelligence()


@st.cache_data(show_spinner=False)
def load_products():
    return build_product_performance()


@st.cache_data(show_spinner=False)
def load_business():
    return build_business_performance()


@st.cache_data(show_spinner=False)
def load_gita_predictions():
    return predict_gita(data=None, save_report=False)


# =========================================================
# 4. REUSABLE HELPERS
# =========================================================

def money(value):
    try:
        number = float(value)
        if pd.isna(number):
            return "—"
        return f"R$ {number:,.2f}"
    except (TypeError, ValueError):
        return "—"


def safe_number(series):
    return pd.to_numeric(series, errors="coerce")


def safe_filename(value):
    result = "".join(
        character if character.isalnum() else "_"
        for character in str(value)
    ).strip("_")
    return result or "report"


def show_table(dataframe, title, key=None):
    if not isinstance(dataframe, pd.DataFrame):
        st.info(f"{title}: no tabular data is available.")
        return

    st.subheader(title)

    if dataframe.empty:
        st.info("No rows are available for this table.")
        return

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download CSV",
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_filename(key or title)}.csv",
        mime="text/csv",
        key=f"download_{safe_filename(key or title)}",
    )


def show_numeric_chart(data, chart_type="bar", value_columns=None):
    if not isinstance(data, pd.DataFrame) or data.empty:
        st.info("No chart data is available.")
        return

    frame = data.copy()

    label_candidates = [
        "month",
        "purchase_month",
        "order_month",
        "year_month",
        "category",
        "Category",
        "Product_Name",
        "Product_ID",
        "product_category_name_english",
        "product_category_name",
        "order_status",
        "reconciliation_status",
    ]

    label_column = next(
        (column for column in label_candidates if column in frame.columns),
        None,
    )

    if value_columns is None:
        numeric_columns = frame.select_dtypes(
            include="number"
        ).columns.tolist()
    else:
        numeric_columns = [
            column for column in value_columns
            if column in frame.columns
            and pd.api.types.is_numeric_dtype(frame[column])
        ]

    if label_column in numeric_columns:
        numeric_columns.remove(label_column)

    if not numeric_columns:
        st.info("No numeric measures are available for this chart.")
        return

    if label_column:
        frame = frame.set_index(label_column)
    else:
        frame.index = range(1, len(frame) + 1)

    chart_data = frame[numeric_columns]

    if chart_type == "line":
        st.line_chart(chart_data)
    else:
        st.bar_chart(chart_data.head(15))


def show_insights(insights):
    if isinstance(insights, str):
        insights = [insights] if insights.strip() else []

    if not insights:
        st.info("No additional insights are available.")
        return

    for insight in insights:
        st.markdown(f"- {insight}")


def run_report(loader, report_name):
    try:
        with st.spinner(f"Preparing {report_name}..."):
            return loader()
    except Exception as exc:
        st.error(f"{report_name} could not be generated.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.caption(
            "Check the source files and terminal output. "
            "The report engine was not changed by this dashboard."
        )
        return None


def read_report_upload(uploaded_file, state_key):
    """Read a report-specific upload and cache it in session state."""
    if uploaded_file is None:
        for suffix in ("", "_signature", "_filename", "_result"):
            st.session_state.pop(f"{state_key}{suffix}", None)
        return None

    raw = uploaded_file.getvalue()

    if len(raw) > MAX_UPLOAD_BYTES:
        for suffix in ("", "_signature", "_filename", "_result"):
            st.session_state.pop(f"{state_key}{suffix}", None)
        st.error("The file exceeds the 25 MB upload limit.")
        return None

    signature = hashlib.sha256(raw).hexdigest()

    if st.session_state.get(f"{state_key}_signature") != signature:
        try:
            dataframe = read_uploaded_dataset(uploaded_file)
        except Exception as exc:
            for suffix in ("", "_signature", "_filename", "_result"):
                st.session_state.pop(f"{state_key}{suffix}", None)
            st.error("The uploaded file could not be read.")
            st.code(f"{type(exc).__name__}: {exc}")
            return None

        st.session_state[state_key] = dataframe
        st.session_state[f"{state_key}_signature"] = signature
        st.session_state[f"{state_key}_filename"] = uploaded_file.name

        # Any prior result based on the previous file must be discarded.
        st.session_state.pop(f"{state_key}_result", None)

    return st.session_state.get(state_key)


def show_validation(dataframe, prefix):
    quality = validate_dataset(dataframe)
    profile = profile_dataset(dataframe)

    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(dataframe):,}")
    b.metric("Columns", f"{len(dataframe.columns):,}")
    c.metric("Missing Cells", f"{int(dataframe.isna().sum().sum()):,}")

    with st.expander("Preview uploaded data", expanded=False):
        st.dataframe(dataframe.head(100), use_container_width=True)

    with st.expander("Data quality report", expanded=False):
        show_table(quality, "Validation Checks", f"{prefix}_quality")

    with st.expander("Column profile", expanded=False):
        show_table(profile, "Column Profile", f"{prefix}_profile")

    return quality, profile


def show_source_status():
    missing = [
        name for name in REQUIRED_DATA_FILES
        if not (DATA_DIR / name).is_file()
    ]

    if missing:
        st.warning(
            "The configured Olist dataset is incomplete. "
            "Reports that require these files may be unavailable."
        )
        with st.expander("Missing Olist source files"):
            for filename in missing:
                st.write(f"- `{filename}`")

    return missing


# =========================================================
# 5. SIDEBAR AND HEADER
# =========================================================

with st.sidebar:
    if LOGO_PATH.is_file():
        # Keep one compact, consistently sized brand mark in the sidebar.
        st.image(str(LOGO_PATH), width=160)
    st.markdown("## SignalMind")
    st.caption("BUSINESS INTELLIGENCE")

    st.markdown("### Appearance")
    st.radio(
        "Theme",
        options=["Light", "Dark"],
        key="sm_theme",
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Workspace")
    st.markdown("- Data Upload & Validation")
    st.markdown("- Sales Report")
    st.markdown("- Revenue Report")
    st.markdown("- Product Performance")
    st.markdown("- Business Performance")
    st.markdown("- Gita Intelligence")

    st.divider()

    st.caption(
        "Gita identifies unusual statistical patterns. "
        "A flag is a review signal, not proof of fraud or loss."
    )

    if st.button("Refresh dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# The logo lives in the sidebar. Keep the main header focused on the workspace.
st.markdown(
    """
    <div class="sm-hero">
        <div class="sm-eyebrow">Business analytics workspace</div>
        <h1>SignalMind</h1>
        <p>
            Validate data, review business performance, and investigate
            unusual patterns with transparent reporting.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
if not LOGO_PATH.is_file():
    st.caption("Logo not found at assets/signalmind_logo.png. Add the file there to display your brand logo.")

st.caption(
    "Each report has its own upload state. Uploading a file for one report "
    "does not automatically make it the data source for another report."
)

missing_files = show_source_status()


# =========================================================
# 6. NAVIGATION
# =========================================================

(
    upload_tab,
    sales_tab,
    revenue_tab,
    product_tab,
    business_tab,
    gita_tab,
) = st.tabs([
    "Data Upload & Validation",
    "Sales Report",
    "Revenue Report",
    "Product Performance",
    "Business Performance",
    "Gita Intelligence",
])


# =========================================================
# 7. GENERAL DATA UPLOAD AND VALIDATION
# =========================================================

with upload_tab:
    st.header("Data Upload & Validation")
    st.write(
        "Use this area to inspect a dataset's structure and basic quality. "
        "This general upload is for inspection only; report-specific uploads "
        "are located inside the corresponding report tabs."
    )

    general_file = st.file_uploader(
        "Choose a dataset to inspect",
        type=["csv", "xlsx", "xls"],
        key="general_validation_upload",
        help="CSV or Excel, maximum 25 MB.",
    )

    if general_file is None:
        st.info("Upload a file here to inspect it.")
    else:
        try:
            if len(general_file.getvalue()) > MAX_UPLOAD_BYTES:
                st.error("The file exceeds the 25 MB upload limit.")
            else:
                signature = hashlib.sha256(
                    general_file.getvalue()
                ).hexdigest()

                if st.session_state.get("general_validation_signature") != signature:
                    # Parse first; only replace the active dataset after success.
                    dataframe = read_uploaded_dataset(general_file)
                    st.session_state["general_validation_signature"] = signature
                    st.session_state["general_validation_df"] = dataframe

                dataframe = st.session_state["general_validation_df"]

                st.success(f"Loaded: {general_file.name}")
                show_validation(dataframe, "general_validation")

                st.caption(
                    "Validation highlights potential data issues. "
                    "Missing values and duplicates are not automatically "
                    "proof that a dataset is invalid."
                )
        except Exception as exc:
            st.error("The dataset could not be inspected.")
            st.code(f"{type(exc).__name__}: {exc}")


# =========================================================
# 8. SALES REPORT — OWN UPLOAD OR EXPLICIT OLIST SOURCE
# =========================================================

with sales_tab:
    st.header("Sales Report")

    sales_source = st.radio(
        "Select the Sales Report source",
        ["Upload a Sales dataset", "Use configured Olist dataset"],
        horizontal=True,
        key="sales_report_source",
    )

    if sales_source == "Upload a Sales dataset":
        sales_file = st.file_uploader(
            "Upload the file for Sales analysis",
            type=["csv", "xlsx", "xls"],
            key="sales_report_upload",
        )

        dataframe = read_report_upload(
            sales_file,
            "sales_report_df",
        )

        if dataframe is None:
            st.info(
                "No Sales file has been selected. Upload a file above "
                "to enable uploaded-data Sales analysis."
            )

        elif dataframe.empty:
            st.warning("The uploaded Sales dataset contains no rows.")

        else:
            filename = st.session_state.get(
                "sales_report_df_filename",
                sales_file.name,
            )

            st.caption(f"Active Sales file: {filename}")
            show_validation(dataframe, "sales_upload")

            numeric_columns = dataframe.select_dtypes(
                include="number"
            ).columns.tolist()

            if not numeric_columns:
                st.error(
                    "No numeric columns were detected. Sales analysis "
                    "requires a numeric sales amount column."
                )
            else:
                preferred = [
                    "sales", "sales_amount", "sale_amount",
                    "revenue", "amount", "total", "price",
                    "item_sales", "total_sales",
                ]
                normalized = {
                    str(column).strip().lower(): column
                    for column in numeric_columns
                }
                default_sales = next(
                    (normalized[name] for name in preferred if name in normalized),
                    numeric_columns[0],
                )

                sales_column = st.selectbox(
                    "Sales amount column",
                    numeric_columns,
                    index=numeric_columns.index(default_sales),
                    key="sales_upload_amount_column",
                )

                columns = dataframe.columns.tolist()
                optional_columns = [None, *columns]

                def pick_optional(label, key, keywords):
                    candidates = [
                        column for column in columns
                        if any(word in str(column).lower() for word in keywords)
                    ]
                    default = candidates[0] if candidates else None
                    options = optional_columns
                    index = options.index(default) if default in options else 0

                    return st.selectbox(
                        label,
                        options,
                        index=index,
                        format_func=lambda value: (
                            "Not selected" if value is None else str(value)
                        ),
                        key=key,
                    )

                c1, c2, c3 = st.columns(3)

                with c1:
                    date_column = pick_optional(
                        "Date column (optional)",
                        "sales_upload_date_column",
                        ["date", "time", "month", "day"],
                    )

                with c2:
                    category_column = pick_optional(
                        "Category column (optional)",
                        "sales_upload_category_column",
                        ["category", "product", "type", "segment"],
                    )

                with c3:
                    order_column = pick_optional(
                        "Order ID column (optional)",
                        "sales_upload_order_column",
                        ["order_id", "order", "transaction_id", "invoice", "receipt"],
                    )

                current_signature = st.session_state.get(
                    "sales_report_df_signature"
                )

                selection = {
                    "sales_column": sales_column,
                    "date_column": date_column,
                    "category_column": category_column,
                    "order_id_column": order_column,
                }

                if st.button(
                    "Analyze Sales",
                    type="primary",
                    use_container_width=True,
                    key="run_uploaded_sales_analysis",
                ):
                    try:
                        with st.spinner("Analyzing uploaded Sales data..."):
                            analysis = analyze_uploaded_sales(
                                dataframe,
                                sales_column=sales_column,
                                date_column=date_column,
                                category_column=category_column,
                                order_id_column=order_column,
                            )

                        st.session_state["sales_report_df_result"] = {
                            "signature": current_signature,
                            "selection": selection,
                            "filename": filename,
                            "analysis": analysis,
                        }
                    except Exception as exc:
                        st.session_state.pop("sales_report_df_result", None)
                        st.error("Sales analysis failed.")
                        st.code(f"{type(exc).__name__}: {exc}")

                saved = st.session_state.get("sales_report_df_result")

                if saved:
                    if (
                        saved.get("signature") != current_signature
                        or saved.get("selection") != selection
                    ):
                        st.session_state.pop("sales_report_df_result", None)
                        saved = None
                        st.info(
                            "The file or selected columns changed. "
                            "Click Analyze Sales again to refresh the result."
                        )

                if saved:
                    analysis = saved["analysis"]
                    summary = analysis.get("summary", {})

                    st.success(f"Sales analysis complete: {saved['filename']}")

                    metric_defs = [
                        ("rows_analyzed", "Rows Analyzed", "count"),
                        ("total_sales", "Total Sales", "money"),
                        ("average_sales_per_row", "Average Sales / Row", "money"),
                        ("distinct_orders", "Distinct Orders", "count"),
                        ("average_sales_per_order", "Average Sales / Order", "money"),
                    ]

                    available = [
                        (key, label, kind)
                        for key, label, kind in metric_defs
                        if key in summary and summary[key] is not None
                    ]

                    if available:
                        metric_cols = st.columns(min(len(available), 4))
                        for index, (key, label, kind) in enumerate(available):
                            value = summary[key]
                            formatted = (
                                money(value) if kind == "money"
                                else f"{int(value):,}" if kind == "count"
                                else str(value)
                            )
                            metric_cols[index % len(metric_cols)].metric(
                                label, formatted
                            )

                    monthly = analysis.get("monthly_sales")
                    categories = analysis.get("category_sales")

                    left, right = st.columns(2)

                    with left:
                        if isinstance(monthly, pd.DataFrame) and not monthly.empty:
                            show_numeric_chart(
                                monthly,
                                "line",
                                ["sales"],
                            )
                            show_table(
                                monthly,
                                "Monthly Sales",
                                "uploaded_sales_monthly",
                            )
                        else:
                            st.info(
                                "Monthly analysis is unavailable. "
                                "Select a suitable date column."
                            )

                    with right:
                        if (
                            isinstance(categories, pd.DataFrame)
                            and not categories.empty
                        ):
                            show_numeric_chart(
                                categories,
                                "bar",
                                ["sales"],
                            )
                            show_table(
                                categories,
                                "Category Sales",
                                "uploaded_sales_categories",
                            )
                        else:
                            st.info(
                                "Category analysis is unavailable. "
                                "Select a suitable category column."
                            )

                    st.subheader("Sales Insights")
                    show_insights(analysis.get("insights", []))

                    st.caption(
                        "These results describe the uploaded data. "
                        "They are not forecasts of future sales."
                    )

    else:
        if missing_files:
            st.info(
                "The configured Olist Sales report cannot run because "
                "one or more required Olist files are missing."
            )
        else:
            st.caption("Source: configured Olist dataset.")
            result = run_report(load_sales, "Olist Sales Report")

            if result is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Delivered Orders",
                    f"{result['total_delivered_orders']:,}",
                )
                c2.metric("Item Sales", money(result["item_sales"]))
                c3.metric("Freight", money(result["freight_total"]))
                c4.metric(
                    "Average Order Value",
                    money(result["average_order_value"]),
                )

                left, right = st.columns(2)

                with left:
                    show_numeric_chart(
                        result["monthly_sales"],
                        "line",
                        ["sales"],
                    )
                    show_table(
                        result["monthly_sales"],
                        "Monthly Sales",
                        "olist_monthly_sales",
                    )

                with right:
                    show_numeric_chart(
                        result["category_sales"],
                        "bar",
                        ["sales"],
                    )
                    show_table(
                        result["category_sales"],
                        "Category Sales",
                        "olist_category_sales",
                    )

                st.subheader("Business Insights")
                show_insights(result.get("insights", []))


# =========================================================
# 9. REVENUE REPORT — OWN UPLOAD OR OLIST PIPELINE
# =========================================================

with revenue_tab:
    st.header("Revenue Report")
    st.caption(
        "Upload a revenue/payment dataset for a general summary, or use the "
        "configured Olist reconciliation pipeline. Uploaded-file analysis is "
        "based only on the columns you select; it does not assume a payment "
        "reconciliation schema."
    )
    revenue_source = st.radio(
        "Revenue Report source",
        ["Upload a Revenue dataset", "Use configured Olist dataset"],
        horizontal=True,
        key="revenue_report_source",
    )

    if revenue_source == "Upload a Revenue dataset":
        revenue_file = st.file_uploader(
            "Upload the file for Revenue analysis",
            type=["csv", "xlsx", "xls"],
            key="revenue_report_upload",
            help="CSV or Excel, maximum 25 MB.",
        )
        dataframe = read_report_upload(revenue_file, "revenue_report_df")
        if dataframe is None:
            st.info("Upload a Revenue CSV/XLSX file to begin analysis.")
        elif dataframe.empty:
            st.warning("The uploaded Revenue dataset contains no rows.")
        else:
            st.caption("Active Revenue file: " + st.session_state.get("revenue_report_df_filename", revenue_file.name))
            show_validation(dataframe, "revenue_upload")
            numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
            if not numeric_columns:
                st.error("No numeric columns were detected. Revenue analysis needs a numeric amount column.")
            else:
                preferred = ["revenue", "payment_value", "amount", "total", "sales", "item_sales", "price"]
                normalized = {str(c).strip().lower(): c for c in numeric_columns}
                default_amount = next((normalized[n] for n in preferred if n in normalized), numeric_columns[0])
                amount_column = st.selectbox("Revenue / payment amount column", numeric_columns, index=numeric_columns.index(default_amount), key="revenue_amount_column")
                id_candidates = [c for c in dataframe.columns if any(token in str(c).lower() for token in ["order_id", "invoice_id", "transaction_id", "payment_id"])]
                id_options = ["(Count rows)"] + id_candidates
                id_column = st.selectbox("Transaction / order identifier (optional)", id_options, key="revenue_id_column")
                amount_series = pd.to_numeric(dataframe[amount_column], errors="coerce")
                valid = amount_series.dropna()
                a, b, c, d = st.columns(4)
                a.metric("Rows", f"{len(dataframe):,}")
                b.metric("Valid amount values", f"{len(valid):,}")
                c.metric("Sum of selected amount", f"{valid.sum():,.2f}")
                d.metric("Average amount", f"{valid.mean():,.2f}" if len(valid) else "N/A")
                if id_column != "(Count rows)":
                    st.metric("Unique transactions / orders", f"{dataframe[id_column].nunique(dropna=True):,}")
                st.caption("These figures summarize the selected amount column. They are not a reconciliation result unless the uploaded data contains the required matched records and fields.")
                st.subheader("Amount distribution")
                st.bar_chart(valid.reset_index(drop=True))
                if id_column != "(Count rows)":
                    grouped = dataframe.assign(_selected_amount=amount_series).groupby(id_column, dropna=False, as_index=False)["_selected_amount"].sum().rename(columns={"_selected_amount": "selected_amount_sum"})
                    show_table(grouped.sort_values("selected_amount_sum", ascending=False).head(100), "Top transactions by selected amount", "revenue_uploaded_transactions")
    else:
        if missing_files:
            st.info("Revenue Report is unavailable until the required Olist files exist.")
        else:
            result = run_report(load_revenue, "Revenue Report")
            if result is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Orders", f"{result['total_orders']:,}")
                c2.metric("Orders Without Items", f"{result['orders_without_items']:,}")
                c3.metric("Orders Without Payments", f"{result['orders_without_payments']:,}")
                c4.metric("Differences Over R$1", f"{result['orders_with_large_differences']:,}")
                st.metric("Delivered Orders With Differences Over R$1", f"{result['delivered_orders_with_large_differences']:,}")
                for key, title in [("reconciliation_summary", "Reconciliation Summary"), ("status_summary", "Totals by Order Status"), ("mismatches", "Orders Requiring Investigation"), ("missing_items_with_payments", "Payments Without Item Records"), ("delivered_differences", "Delivered Orders With Large Differences")]:
                    show_table(result.get(key), title, f"revenue_{key}")
                st.warning("A payment difference or missing record is an investigation flag, not proof of a revenue loss.")


# =========================================================
# 10. PRODUCT PERFORMANCE — OWN UPLOAD OR OLIST PIPELINE
# =========================================================

with product_tab:
    st.header("Product Performance")

    product_source = st.radio(
        "Select the Product Performance source",
        ["Upload a Product dataset", "Use configured Olist dataset"],
        horizontal=True,
        key="product_report_source",
    )

    if product_source == "Upload a Product dataset":
        product_file = st.file_uploader(
            "Upload the file for Product Performance",
            type=["csv", "xlsx", "xls"],
            key="product_report_upload",
            help="For example: Product_Performance_Demo.csv",
        )

        dataframe = read_report_upload(
            product_file,
            "product_report_df",
        )

        if dataframe is None:
            st.info(
                "No Product Performance file has been selected. "
                "Upload a file above to display product-specific results."
            )

        elif dataframe.empty:
            st.warning("The uploaded Product dataset contains no rows.")

        else:
            filename = st.session_state.get(
                "product_report_df_filename",
                product_file.name,
            )

            st.caption(f"Active Product file: {filename}")
            show_validation(dataframe, "product_upload")

            # Normalize column names only for matching; preserve originals.
            column_map = {
                str(column).strip().lower().replace(" ", "_"): column
                for column in dataframe.columns
            }

            def find_column(*names):
                for name in names:
                    if name in column_map:
                        return column_map[name]
                return None

            product_id_col = find_column(
                "product_id", "product", "sku", "item_id"
            )
            product_name_col = find_column(
                "product_name", "name", "product"
            )
            category_col = find_column(
                "category", "product_category", "product_category_name"
            )
            units_col = find_column(
                "units_sold", "quantity_sold", "quantity", "units", "qty"
            )
            price_col = find_column(
                "unit_price", "price", "selling_price"
            )
            sales_col = find_column(
                "sales", "revenue", "sales_revenue", "total_sales"
            )
            returns_col = find_column(
                "returns", "return_count", "returned_units"
            )

            numeric_columns = dataframe.select_dtypes(
                include="number"
            ).columns.tolist()

            st.subheader("Configure Product Analysis")

            a, b, c = st.columns(3)

            with a:
                units_col = st.selectbox(
                    "Units sold (optional)",
                    [None, *dataframe.columns],
                    index=(
                        [None, *dataframe.columns].index(units_col)
                        if units_col in dataframe.columns else 0
                    ),
                    format_func=lambda value: (
                        "Not selected" if value is None else str(value)
                    ),
                    key="product_units_column",
                )

            with b:
                sales_col = st.selectbox(
                    "Sales amount (optional)",
                    [None, *numeric_columns],
                    index=(
                        [None, *numeric_columns].index(sales_col)
                        if sales_col in numeric_columns else 0
                    ),
                    format_func=lambda value: (
                        "Not selected" if value is None else str(value)
                    ),
                    key="product_sales_column",
                )

            with c:
                returns_col = st.selectbox(
                    "Returns (optional)",
                    [None, *dataframe.columns],
                    index=(
                        [None, *dataframe.columns].index(returns_col)
                        if returns_col in dataframe.columns else 0
                    ),
                    format_func=lambda value: (
                        "Not selected" if value is None else str(value)
                    ),
                    key="product_returns_column",
                )

            category_col = st.selectbox(
                "Category column (optional)",
                [None, *dataframe.columns],
                index=(
                    [None, *dataframe.columns].index(category_col)
                    if category_col in dataframe.columns else 0
                ),
                format_func=lambda value: (
                    "Not selected" if value is None else str(value)
                ),
                key="product_category_column",
            )

            product_label_col = product_name_col or product_id_col
            if product_label_col is None:
                product_label_col = st.selectbox(
                    "Product identifier/name column",
                    dataframe.columns.tolist(),
                    key="product_label_column",
                )

            selection = {
                "units": units_col,
                "sales": sales_col,
                "returns": returns_col,
                "category": category_col,
                "label": product_label_col,
            }

            if st.button(
                "Analyze Product Performance",
                type="primary",
                use_container_width=True,
                key="run_uploaded_product_analysis",
            ):
                try:
                    result_df = dataframe.copy()

                    # Convert selected measures safely to numeric values.
                    for selected in [units_col, sales_col, returns_col]:
                        if selected is not None:
                            result_df[selected] = pd.to_numeric(
                                result_df[selected],
                                errors="coerce",
                            )

                    st.session_state["product_report_df_result"] = {
                        "signature": st.session_state.get(
                            "product_report_df_signature"
                        ),
                        "selection": selection,
                        "filename": filename,
                        "data": result_df,
                    }
                except Exception as exc:
                    st.session_state.pop("product_report_df_result", None)
                    st.error("Product analysis could not be prepared.")
                    st.code(f"{type(exc).__name__}: {exc}")

            saved = st.session_state.get("product_report_df_result")

            if saved and (
                saved.get("signature")
                != st.session_state.get("product_report_df_signature")
                or saved.get("selection") != selection
            ):
                st.session_state.pop("product_report_df_result", None)
                saved = None
                st.info(
                    "The file or selected columns changed. Click "
                    "Analyze Product Performance again."
                )

            if saved:
                result_df = saved["data"]
                st.success(f"Product analysis complete: {saved['filename']}")

                units = (
                    safe_number(result_df[units_col])
                    if units_col is not None
                    else None
                )
                sales = (
                    safe_number(result_df[sales_col])
                    if sales_col is not None
                    else None
                )
                returns = (
                    safe_number(result_df[returns_col])
                    if returns_col is not None
                    else None
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Product Records", f"{len(result_df):,}")

                if units is not None:
                    m2.metric(
                        "Units Sold",
                        f"{int(units.sum(skipna=True)):,}",
                    )
                else:
                    m2.metric("Units Sold", "Not available")

                if sales is not None:
                    m3.metric("Total Sales", money(sales.sum(skipna=True)))
                else:
                    m3.metric("Total Sales", "Not available")

                if returns is not None:
                    m4.metric(
                        "Recorded Returns",
                        f"{int(returns.sum(skipna=True)):,}",
                    )
                else:
                    m4.metric("Recorded Returns", "Not available")

                st.divider()

                if sales_col is not None:
                    ranked = result_df.copy()
                    ranked[sales_col] = safe_number(ranked[sales_col])
                    ranked = ranked.sort_values(
                        sales_col,
                        ascending=False,
                        na_position="last",
                    )

                    show_numeric_chart(
                        ranked.head(10).set_index(product_label_col)
                        if product_label_col in ranked.columns
                        else ranked.head(10),
                        "bar",
                        [sales_col],
                    )

                    show_table(
                        ranked.head(10),
                        "Top 10 Products by Sales",
                        "uploaded_product_top_sales",
                    )

                if category_col is not None and sales_col is not None:
                    category_data = result_df.copy()
                    category_data[sales_col] = safe_number(
                        category_data[sales_col]
                    )

                    category_summary = (
                        category_data.groupby(
                            category_col,
                            dropna=False,
                        )[sales_col]
                        .sum(min_count=1)
                        .reset_index()
                        .sort_values(
                            sales_col,
                            ascending=False,
                            na_position="last",
                        )
                    )

                    show_numeric_chart(
                        category_summary,
                        "bar",
                        [sales_col],
                    )
                    show_table(
                        category_summary,
                        "Sales by Category",
                        "uploaded_product_category_sales",
                    )

                if units_col is not None:
                    unit_summary = result_df.copy()
                    unit_summary[units_col] = safe_number(
                        unit_summary[units_col]
                    )
                    unit_summary = unit_summary.sort_values(
                        units_col,
                        ascending=False,
                        na_position="last",
                    )
                    show_table(
                        unit_summary.head(10),
                        "Top 10 Products by Units Sold",
                        "uploaded_product_top_units",
                    )

                if returns_col is not None:
                    return_summary = result_df.copy()
                    return_summary[returns_col] = safe_number(
                        return_summary[returns_col]
                    )
                    return_summary = return_summary.sort_values(
                        returns_col,
                        ascending=False,
                        na_position="last",
                    )
                    show_table(
                        return_summary.head(10),
                        "Products With Most Recorded Returns",
                        "uploaded_product_returns",
                    )

                show_table(
                    result_df,
                    "Uploaded Product Dataset",
                    "uploaded_product_full_data",
                )

                st.caption(
                    "These results summarize the uploaded file and the "
                    "columns you selected. They do not predict future sales."
                )

    else:
        st.caption("Source: configured Olist dataset.")

        if missing_files:
            st.info(
                "The Olist Product Performance report cannot run because "
                "one or more required source files are missing."
            )
        else:
            result = run_report(load_products, "Olist Product Performance")

            if result is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Delivered Item Sales",
                    money(result["delivered_item_sales"]),
                )
                c2.metric(
                    "Delivered Item Records",
                    f"{result['delivered_item_rows']:,}",
                )
                c3.metric(
                    "Delivered Orders",
                    f"{result['delivered_order_count']:,}",
                )

                product_data = result["product_performance"]
                category_data = result["category_performance"]

                left, right = st.columns(2)

                with left:
                    show_numeric_chart(product_data, "bar")
                    show_table(
                        product_data.head(10),
                        "Top 10 Products",
                        "olist_top_products",
                    )

                with right:
                    show_numeric_chart(category_data, "bar")
                    show_table(
                        category_data.head(10),
                        "Top 10 Categories",
                        "olist_top_categories",
                    )

                c1, c2 = st.columns(2)
                c1.metric(
                    "Unmatched Order Items",
                    f"{result['unmatched_order_items']:,}",
                )
                c2.metric(
                    "Unmatched Product Rows",
                    f"{result['unmatched_product_rows']:,}",
                )


# =========================================================
# 11. BUSINESS PERFORMANCE — OWN UPLOAD OR OLIST PIPELINE
# =========================================================

with business_tab:
    st.header("Business Performance")
    st.caption(
        "Upload a business dataset to inspect overall quality and compare a "
        "numeric measure across a category, or use the configured Olist report modules."
    )
    business_source = st.radio(
        "Business Performance source",
        ["Upload a Business dataset", "Use configured Olist dataset"],
        horizontal=True,
        key="business_report_source",
    )

    if business_source == "Upload a Business dataset":
        business_file = st.file_uploader(
            "Upload the file for Business Performance analysis",
            type=["csv", "xlsx", "xls"],
            key="business_report_upload",
            help="CSV or Excel, maximum 25 MB.",
        )
        dataframe = read_report_upload(business_file, "business_report_df")
        if dataframe is None:
            st.info("Upload a Business Performance CSV/XLSX file to begin analysis.")
        elif dataframe.empty:
            st.warning("The uploaded Business Performance dataset contains no rows.")
        else:
            st.caption("Active Business file: " + st.session_state.get("business_report_df_filename", business_file.name))
            show_validation(dataframe, "business_upload")
            numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
            if not numeric_columns:
                st.info("The dataset has no numeric columns for metric comparisons. You can still review its preview and data-quality report above.")
            else:
                metric_column = st.selectbox("Business metric to analyze", numeric_columns, key="business_metric_column")
                category_candidates = [c for c in dataframe.columns if c not in numeric_columns or not pd.api.types.is_numeric_dtype(dataframe[c])]
                category_options = ["(No grouping — overall summary)"] + category_candidates
                category_column = st.selectbox("Group by a category / business dimension (optional)", category_options, key="business_category_column")
                metric_series = pd.to_numeric(dataframe[metric_column], errors="coerce")
                valid = metric_series.dropna()
                a, b, c, d = st.columns(4)
                a.metric("Rows", f"{len(dataframe):,}")
                b.metric("Valid metric values", f"{len(valid):,}")
                c.metric("Sum", f"{valid.sum():,.2f}")
                d.metric("Average", f"{valid.mean():,.2f}" if len(valid) else "N/A")
                if category_column == "(No grouping — overall summary)":
                    st.subheader("Selected metric distribution")
                    st.bar_chart(valid.reset_index(drop=True))
                else:
                    grouped = dataframe[[category_column]].copy()
                    grouped[metric_column] = metric_series
                    summary = grouped.groupby(category_column, dropna=False)[metric_column].agg(["count", "sum", "mean", "min", "max"]).reset_index()
                    summary = summary.sort_values("sum", ascending=False, na_position="last")
                    st.subheader("Business metric by " + str(category_column))
                    chart_data = summary.set_index(category_column)["sum"].head(30)
                    st.bar_chart(chart_data)
                    show_table(summary, "Grouped Business Performance", "business_uploaded_group_summary")
                st.caption("This is a general uploaded-data summary. Business-specific KPIs depend on the meaning and structure of your columns; no unsupported business conclusions are inferred automatically.")
    else:
        if missing_files:
            st.info("Business Performance is unavailable until the required Olist source files are present.")
        else:
            result = run_report(load_business, "Business Performance")

        if result is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Orders", f"{result['total_orders']:,}")
            c2.metric(
                "Delivered Orders",
                f"{result['total_delivered_orders']:,}",
            )
            c3.metric(
                "Delivered Order Share",
                f"{result['delivered_order_share_percent']:.2f}%",
            )
            c4.metric("Item Sales", money(result["item_sales"]))

            left, right = st.columns(2)

            with left:
                st.subheader("Revenue Reconciliation")
                st.metric(
                    "Orders Without Items",
                    f"{result['orders_without_items']:,}",
                )
                st.metric(
                    "Orders Without Payments",
                    f"{result['orders_without_payments']:,}",
                )
                st.metric(
                    "Orders With Large Differences",
                    f"{result['orders_with_large_differences']:,}",
                )
                st.metric(
                    "Large-Difference Rate",
                    f"{result['large_difference_rate_percent']:.2f}%",
                )

            with right:
                st.subheader("Product and Freight Indicators")
                st.metric("Freight Total", money(result["freight_total"]))
                st.metric(
                    "Unmatched Order Items",
                    f"{result['unmatched_order_items']:,}",
                )
                st.metric(
                    "Unmatched Product Rows",
                    f"{result['unmatched_product_rows']:,}",
                )

                if result.get("top_category"):
                    st.write("**Top Category**")
                    st.json(result["top_category"])

                if result.get("top_product"):
                    st.write("**Top Product**")
                    st.json(result["top_product"])

            show_table(
                result["reconciliation_summary"],
                "Reconciliation Summary",
                "business_reconciliation",
            )
            show_table(
                result["status_summary"],
                "Order Status Summary",
                "business_order_status",
            )

            st.subheader("Business Insights")
            show_insights(result.get("insights", []))


# =========================================================
# 12. GITA INTELLIGENCE — SAVED OLIST MODEL OR UPLOADED DATA
# =========================================================

with gita_tab:
    st.header("Gita Intelligence")
    st.write(
        "Choose the saved Olist model for the existing order-level demo, "
        "or run an exploratory Isolation Forest on a general uploaded "
        "business dataset."
    )

    gita_source = st.radio(
        "Choose Gita's data source",
        ["Analyze an uploaded dataset", "Use saved Olist model"],
        horizontal=True,
        key="gita_data_source",
    )

    if gita_source == "Analyze an uploaded dataset":
        st.markdown("#### Upload a business dataset")
        st.caption(
            "This path fits an exploratory model to the selected numeric "
            "columns in this upload. It does not use or replace the saved "
            "Olist-trained model."
        )

        gita_file = st.file_uploader(
            "Upload CSV or Excel for Gita",
            type=["csv", "xlsx", "xls"],
            key="gita_uploaded_file",
            help="Maximum file size: 25 MB.",
        )

        uploaded_gita_df = read_report_upload(gita_file, "gita_uploaded_df")

        if uploaded_gita_df is None:
            st.info("Upload a CSV/XLSX file to begin Gita's exploratory analysis.")
        elif uploaded_gita_df.empty:
            st.warning("The uploaded dataset contains no rows.")
        else:
            st.caption(
                "Active file: "
                + st.session_state.get("gita_uploaded_df_filename", gita_file.name)
            )
            show_validation(uploaded_gita_df, "gita_uploaded_validation")
            render_uploaded_gita_tab(uploaded_gita_df)

    else:
        st.markdown("#### Saved Olist model")
        st.caption(
            "This path uses the saved Gita model trained on Olist order, item, "
            "freight, and payment features. It does not analyze arbitrary "
            "uploaded columns."
        )

        st.header("Gita Intelligence")

        st.write(
            "Gita uses the saved Isolation Forest model to flag unusual "
            "combinations of order and financial features in eligible Olist data."
        )

        st.warning(
            "An anomaly is a statistical review signal, not a confirmed error, "
            "fraud finding, or financial loss. Anomaly scores are not probabilities."
        )

        if missing_files:
            st.info(
                "Gita requires the configured Olist source data. "
                "The uploaded Product Performance file is not used as a substitute."
            )
        elif not MODEL_PATH.is_file():
            st.error(
                f"Saved Gita model not found: `{MODEL_PATH}`. "
                "Check that the model file exists."
            )
        else:
            predictions = run_report(load_gita_predictions, "Gita Intelligence")

            if predictions is not None:
                if not isinstance(predictions, pd.DataFrame) or predictions.empty:
                    st.info("Gita returned no eligible orders.")
                elif "gita_is_anomaly" not in predictions.columns:
                    st.error(
                        "Gita output is missing the required "
                        "`gita_is_anomaly` column."
                    )
                else:
                    # Handle common boolean representations safely.
                    raw_flag = predictions["gita_is_anomaly"]

                    if pd.api.types.is_bool_dtype(raw_flag):
                        anomaly_mask = raw_flag.fillna(False)
                    else:
                        anomaly_mask = (
                            raw_flag.astype(str)
                            .str.strip()
                            .str.lower()
                            .isin(["true", "1", "yes", "-1", "anomaly"])
                        )

                    flagged = predictions.loc[anomaly_mask].copy()
                    total = len(predictions)
                    anomaly_count = len(flagged)
                    normal_count = total - anomaly_count
                    flag_rate = (
                        anomaly_count / total * 100 if total else 0.0
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Orders Scored", f"{total:,}")
                    c2.metric("Unusual Patterns", f"{anomaly_count:,}")
                    c3.metric("Not Flagged", f"{normal_count:,}")
                    c4.metric("Flag Rate", f"{flag_rate:.2f}%")

                    if "gita_model_version" in predictions.columns:
                        versions = (
                            predictions["gita_model_version"]
                            .dropna()
                            .unique()
                        )
                        if len(versions):
                            st.caption(f"Model version: {versions[0]}")

                    display_columns = [
                        column for column in [
                            "order_id",
                            "order_status",
                            "item_sales",
                            "freight",
                            "freight_ratio",
                            "payment_total",
                            "difference",
                            "gita_anomaly_score",
                            "gita_result",
                        ]
                        if column in flagged.columns
                    ]

                    st.subheader("Flagged Orders")

                    if flagged.empty:
                        st.success("No orders were flagged in this evaluation.")
                    elif display_columns:
                        show_table(
                            flagged[display_columns].head(100),
                            "Top 100 Flagged Orders",
                            "gita_flagged_orders",
                        )
                    else:
                        st.warning(
                            "Flagged rows exist, but recognized display columns "
                            "were not returned by the prediction module."
                        )

                    if "gita_anomaly_score" in predictions.columns:
                        scores = pd.to_numeric(
                            predictions["gita_anomaly_score"],
                            errors="coerce",
                        ).dropna()

                        st.subheader("Anomaly Score Distribution")

                        if scores.nunique() > 1:
                            distribution = pd.cut(
                                scores,
                                bins=20,
                                duplicates="drop",
                            ).value_counts(sort=False)

                            distribution.index = distribution.index.astype(str)
                            distribution.name = "Order Count"
                            st.bar_chart(distribution)
                        else:
                            st.info(
                                "A score distribution chart is not useful "
                                "because the scores do not vary."
                            )

                    st.download_button(
                        "Download All Gita Predictions",
                        data=predictions.to_csv(index=False).encode("utf-8"),
                        file_name="gita_predictions.csv",
                        mime="text/csv",
                        key="download_gita_predictions",
                    )

                    st.caption(
                        "Gita uses the saved model and does not retrain it "
                        "when this dashboard is opened."
                    )

# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "SignalMind | Business analytics, data validation, "
    "sales reporting, reconciliation, and statistical anomaly review"
)
