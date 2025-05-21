import streamlit as st
import polars as pl
import plotly.express as px
from pathlib import Path
import re
import plotly.graph_objects as go


# ---------- CONFIG ----------
st.set_page_config(page_title="KPI Dashboard", layout="wide")

# ---------- TITLE ----------
st.title("📊 KPI Dashboard")


def sbt(input_s: str) -> dict:
    parts = re.split(r'[_-]', input_s)
    if len(parts) < 6:
        return {"Site": None, "Sector": None, "Band": None, "Type": None}
    
    sector = "_".join([parts[4], parts[5]]) if len(str(parts[4])) < 7 else parts[4]
    band = "_".join([parts[2], parts[5]])
    type_ = "_".join([parts[2], "TB"]) if parts[5] in ["L", "M", "N", "O"] else parts[2]
    site = sector[:6] if sector else None
    
    return {"Site": site, "Sector": sector, "Band": band, "Type": type_}

def read_file(filepath: Path) -> pl.DataFrame:
    if filepath.suffix == ".csv":
        return pl.read_csv(filepath)
    elif filepath.suffix == ".xlsx":
        return pl.read_excel(filepath)
    else:
        return None

def process_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    df = df.rename({col: col.strip() for col in df.columns})

    if "_UNNAMED_1" in df.columns:
        df = df.rename({"_UNNAMED_1": "Date"})

    if "Date" in df.columns:
        df = df.with_columns(pl.col("Date").dt.strftime("%d-%b-%y").alias("Date"))

    if "Short name" in df.columns:
        df = df.with_columns(
            pl.col("Short name").map_elements(sbt, return_dtype=pl.Struct).alias("fields")
        ).unnest("fields")

    return df

def combine_files(folder_path: str) -> pl.DataFrame:
    folder = Path(folder_path)
    all_files = list(folder.glob("*.csv")) + list(folder.glob("*.xlsx"))
    dfs = []
    for file in all_files:
        try:
            df = read_file(file)
            if df is not None:
                df = process_dataframe(df)
                dfs.append(df)
        except Exception as e:
            print(f"⚠ Skipped {file.name} due to error: {e}")
    if not dfs:
        raise ValueError("No valid files found to process.")
    final_df = pl.concat(dfs, how="vertical")
    if "__UNNAMED__1" in final_df.columns:
        final_df = final_df.rename({"__UNNAMED__1": "Date"})
    if "Date" in final_df.columns:
        final_df = final_df.with_columns(pl.col("Date").alias("New Date"))
        final_df = final_df.with_columns(pl.col("Date").dt.strftime("%d-%b-%y").alias("Date"))
    final_df = final_df.with_columns(
        pl.col("New Date")
    ).sort("New Date")
    return final_df

def get_kpi_info_by_name(kpi_list, target_kpi_name) -> dict:
    for entry in kpi_list:
        if entry.get("KPI_Name") == target_kpi_name:
            return {k: v for k, v in entry.items() if k != "KPI_Name"}
    return None

def pivot_plot_threshold_helper_func(df, kpi_col):
    dict_output = st.session_state.dict_output
    result = get_kpi_info_by_name(dict_output, kpi_col)

    # ⏱️ Date range
    min_date = df.select(pl.col("New Date").min()).item()
    max_date = df.select(pl.col("New Date").max()).item()

    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Start Date", min_date.strftime("%d-%b-%y"))
    col2.metric("🕒 End Date", max_date.strftime("%d-%b-%y"))
    col3.metric("🔢 Data Points", len(df))

    with st.expander("🔧 Advanced Filters", expanded=False):
        site_name = st.text_input("🏗️ Filter by Site Name (e.g., DHUL92):", value="")

        # Date inputs
        date_range = st.date_input(
            "📅 Select Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    from_date, to_date = date_range
    filter_conditions = (pl.col("New Date") >= from_date) & (pl.col("New Date") <= to_date)
    if site_name.strip():
        filter_conditions &= pl.col("Site").str.contains(site_name.strip(), literal=False)

    df = df.filter(filter_conditions)
    layer_flag="no"
    with st.expander("🎯 KPI Configuration", expanded=False):
        if result["Threshold"] is not None:
            threshold = st.slider("📊 Set Threshold", min_value=90.0, max_value=100.0,
                                value=float(result["Threshold"]), step=0.1)
            comparison_sign = st.selectbox("🔁 Comparison Operator", options=["<", ">"],
                                        index=0 if result["Sign"] != ">" else 1)
            
        if result.get("Layer_wise", "no") == "yes":
            layer_flag = st.selectbox("🧬 Enable Layer-wise Classification?", options=["Yes", "No"], index=0)

    st.markdown("### 📈 KPI Trend Overview")
    is_mean = result["Function"] == "mean"
    selected_layer = None
    if is_mean:
        if layer_flag.strip().lower() == "yes":
            avg_df = (
                df.group_by("New Date", "Type")
                .agg(pl.col(kpi_col).mean().round(2).alias(f"Avg_{kpi_col}"))
                .sort("New Date")
                .with_columns(pl.col("New Date").dt.strftime("%d-%b-%y").alias("Date"))
            )
            show_all_layers = st.checkbox("Show all layers", value=True)

            if show_all_layers:
                fig = px.line(
                    avg_df,
                    x="Date",
                    y=f"Avg_{kpi_col}",
                    color="Type",
                    markers=True,
                    labels={"Date": "Date", "Type": "Layer"},
                    text=f"Avg_{kpi_col}"
                )
            else:    
                selected_layer = st.selectbox("Select Layer:", options=avg_df["Type"].unique().to_list())
                filtered_df = avg_df.filter(pl.col("Type") == selected_layer)
                df = df.filter(pl.col("Type") == selected_layer)
                fig = px.line(
                    filtered_df,
                    x="Date",
                    y=f"Avg_{kpi_col}",
                    color="Type",
                    markers=True,
                    labels={"Date": "Date", "Type": "Layer"},
                    text=f"Avg_{kpi_col}"
                )
        else:
            avg_df = (
                df.group_by("New Date")
                .agg(pl.col(kpi_col).mean().round(2).alias(f"Avg_{kpi_col}"))
                .sort("New Date")
                .with_columns(pl.col("New Date").dt.strftime("%d-%b-%y").alias("Date"))
            )
            fig = px.line(
                avg_df,
                x="Date",
                y=f"Avg_{kpi_col}",
                markers=True,
                labels={"Date": "Date"},
                text=f"Avg_{kpi_col}"
            )
    else:
        avg_df = (
            df.group_by("New Date")
            .agg(pl.col(kpi_col).sum().round(2).alias(f"Sum_{kpi_col}"))
            .sort("New Date")
            .with_columns(pl.col("New Date").dt.strftime("%d-%b-%y").alias("Date"))
        )
        fig = px.line(
            avg_df,
            x="Date",
            y=f"Sum_{kpi_col}",
            markers=True,
            labels={"Date": "Date"},
            text=f"Sum_{kpi_col}"
        )

    fig.update_layout(showlegend=True, hovermode="x unified",title={
        'text': f"KPI Trend: {kpi_col}",
        'x': 0.5,
        'xanchor': 'center',
        'font': dict(size=20)
    },
    xaxis=dict(title="Date"),
    yaxis=dict(title=f"{kpi_col} Value"),
    margin=dict(l=40, r=40, t=60, b=40),
    # template="plotly_white"
    )
    fig.update_traces(
        textposition="top right",
        texttemplate="%{text:.2f}",
        marker=dict(size=10)
    )
    events = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    # 🚨 Threshold failure summary (if applicable)
    if is_mean and result["Threshold"] is not None:
        fail_flag_expr = (pl.col(kpi_col) < threshold) if comparison_sign == "<" else (pl.col(kpi_col) > threshold)
        fail_df = df.with_columns([fail_flag_expr.cast(pl.Int64).alias("fail_flag")])

        summary_df = (
            fail_df.group_by("New Date")
            .agg([
                pl.len().alias("Total Cells Evaluated"),
                pl.sum("fail_flag").alias("Failing Cells")
            ])
            .with_columns([
                (pl.col("Failing Cells") / pl.col("Total Cells Evaluated") * 100)
                .round(2)
                .alias("Failure Rate %"),
                pl.col("New Date").dt.strftime("%d-%b-%y").alias("Date")
            ])
            .drop("New Date")
        )
        summary_df = summary_df.select(["Date"] + [col for col in summary_df.columns if col != "Date"])
        st.markdown("### 📊 Failure Summary")
        st.dataframe(summary_df, use_container_width=True)

        # 📌 On plot click: show failed cells
        if events and events.selection and events.selection.points:
            clicked_date = events.selection.points[0]['x']
            st.markdown(f"### 🔎 Failure Breakdown for `{clicked_date}`")
            failed_cells = (
                df.filter(pl.col("Date") == clicked_date)
                .filter(fail_flag_expr)
                .select(["Short name", "Date", kpi_col])
                .sort(kpi_col, descending=False)
            )
            st.dataframe(failed_cells, use_container_width=True)

# ---------- FILE UPLOAD ----------
if "data" not in st.session_state:
    std_temp = st.file_uploader("📁 Upload Your Standard Template", type=["csv", "xlsx"])
    if std_temp:
        try:
            if std_temp.name.endswith(".csv"):
                std_df = pl.read_csv(std_temp)
            else:
                std_df = pl.read_excel(std_temp)
            dict_output = std_df.to_dicts()
            grouped_kpis = {
                category: std_df.filter(pl.col("KPI_Category") == category)["KPI_Name"].to_list()
                for category in std_df["KPI_Category"].unique().to_list()
            }
            st.session_state.dict_output=dict_output
            st.session_state.grouped_kpis = grouped_kpis
        except Exception as e:
            print(f"❌ Error: {e}")
    
    folder_input = st.text_input("📂 Enter the full path to the folder containing .csv/.xlsx files: ").strip()
    # print(folder_input)
    try:
        df = combine_files(folder_input)
        st.session_state.data = df
        st.success("✅ File uploaded and processed successfully!")
    except Exception as err:
        print(f"❌ Error: {err}")
    

# ---------- MAIN DASHBOARD ----------
if "data" in st.session_state:
    df = st.session_state.data
    grouped_kpis=st.session_state.grouped_kpis
    st.sidebar.title("🧭 Navigation")
    selected_kpi = st.sidebar.selectbox("Select Category", list(grouped_kpis.keys()))
    selected_plot = st.sidebar.radio("📌 Select KPI", grouped_kpis[selected_kpi])


    # --- Main KPI Rendering Logic ---
    st.markdown(f"### 📍 {selected_kpi} - {selected_plot}")
    try:
            pivot_plot_threshold_helper_func(df,selected_plot)
            
                  
    except Exception as e:
        if e=="list index out of range":
            st.error("❌ No data available for the selected date. Please select a different date.")
