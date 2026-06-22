import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI EV Transportation Intelligence Platform",
    page_icon="🚗",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}

div[data-testid="metric-container"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.15);
}

h1, h2, h3 {
    color: #1f4e79;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("electric_vehicles_dataset.csv")

model = joblib.load("ev_sales_model.pkl")

# ==========================
# TITLE
# ==========================

st.title("🚗 AI-Powered EV Transportation Intelligence Platform")

st.markdown("""
### Smart Analytics, Market Intelligence & AI Sales Forecasting
""")

# ==========================
# SIDEBAR FILTERS
# ==========================

st.sidebar.header("🔍 Dashboard Filters")

manufacturer = st.sidebar.selectbox(
    "Manufacturer",
    ["All"] + sorted(df["Manufacturer"].unique().tolist())
)

battery_type = st.sidebar.selectbox(
    "Battery Type",
    ["All"] + sorted(df["Battery_Type"].unique().tolist())
)

if manufacturer != "All":
    df = df[df["Manufacturer"] == manufacturer]

if battery_type != "All":
    df = df[df["Battery_Type"] == battery_type]

# ==========================
# KPI SECTION
# ==========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🚗 Total Vehicles",
        len(df)
    )

with col2:
    st.metric(
        "🔋 Avg Battery",
        round(df["Battery_Capacity_kWh"].mean(), 1)
    )

with col3:
    st.metric(
        "⚡ Avg Range",
        round(df["Range_km"].mean(), 1)
    )

with col4:
    st.metric(
        "📈 Total Sales",
        f"{int(df['Units_Sold_2024'].sum()):,}"
    )

st.divider()

# ==========================
# SALES ANALYSIS
# ==========================

st.subheader("📈 EV Sales by Manufacturer")

sales_df = (
    df.groupby("Manufacturer")["Units_Sold_2024"]
    .sum()
    .reset_index()
)

fig1 = px.bar(
    sales_df,
    x="Manufacturer",
    y="Units_Sold_2024",
    color="Manufacturer",
    title="Manufacturer Market Performance"
)

st.plotly_chart(fig1, use_container_width=True)

# ==========================
# RANGE ANALYSIS
# ==========================

st.subheader("🔋 EV Range Distribution")

fig2 = px.histogram(
    df,
    x="Range_km",
    nbins=20,
    title="Vehicle Range Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

# ==========================
# BATTERY VS RANGE
# ==========================

st.subheader("⚡ Battery Capacity vs Range")

fig3 = px.scatter(
    df,
    x="Battery_Capacity_kWh",
    y="Range_km",
    color="Manufacturer",
    size="Units_Sold_2024",
    hover_data=["Model"]
)

st.plotly_chart(fig3, use_container_width=True)

# ==========================
# BATTERY TYPE ANALYSIS
# ==========================

st.subheader("🔌 Battery Technology Distribution")

battery_df = (
    df["Battery_Type"]
    .value_counts()
    .reset_index()
)

battery_df.columns = [
    "Battery_Type",
    "Count"
]

fig4 = px.pie(
    battery_df,
    names="Battery_Type",
    values="Count"
)

st.plotly_chart(fig4, use_container_width=True)

# ==========================
# COUNTRY ANALYSIS
# ==========================

st.subheader("🌍 Manufacturing Country Analysis")

country_df = (
    df.groupby("Country_of_Manufacture")["Units_Sold_2024"]
    .sum()
    .reset_index()
)

fig5 = px.bar(
    country_df,
    x="Country_of_Manufacture",
    y="Units_Sold_2024",
    color="Country_of_Manufacture"
)

st.plotly_chart(fig5, use_container_width=True)

# ==========================
# AI INSIGHTS
# ==========================

st.subheader("🤖 AI Market Insights")

best_range = df.loc[df["Range_km"].idxmax()]

st.success(
    f"Highest Range EV: {best_range['Manufacturer']} {best_range['Model']} ({best_range['Range_km']} km)"
)

st.info(
    f"Most Popular Battery Technology: {df['Battery_Type'].mode()[0]}"
)

st.warning(
    f"Average Warranty Offered: {round(df['Warranty_Years'].mean(),1)} years"
)

# ==========================
# AI PREDICTION SECTION
# ==========================

st.divider()

st.subheader("🚀 AI EV Sales Demand Predictor")

colA, colB = st.columns(2)

with colA:

    battery = st.slider(
        "Battery Capacity (kWh)",
        20,
        200,
        80
    )

    ev_range = st.slider(
        "Range (km)",
        100,
        1000,
        500
    )

    charge_time = st.slider(
        "Charge Time (Hours)",
        1,
        24,
        8
    )

    price = st.number_input(
        "Price (USD)",
        10000,
        200000,
        50000
    )

with colB:

    autonomous = st.slider(
        "Autonomous Level",
        0,
        5,
        2
    )

    safety = st.slider(
        "Safety Rating",
        1,
        5,
        4
    )

    warranty = st.slider(
        "Warranty Years",
        1,
        15,
        8
    )

if st.button("🤖 Predict Future Sales"):

    features = np.array([[
        battery,
        ev_range,
        charge_time,
        price,
        autonomous,
        safety,
        warranty
    ]])

    prediction = model.predict(features)[0]

    st.success(
        f"Predicted Units Sold: {int(prediction):,}"
    )

    if prediction > 100000:
        st.success("🟢 High Market Demand")
    elif prediction > 50000:
        st.warning("🟡 Moderate Market Demand")
    else:
        st.error("🔴 Low Market Demand")

# ==========================
# DATASET
# ==========================

st.subheader("📄 EV Dataset Explorer")

st.dataframe(df, use_container_width=True)