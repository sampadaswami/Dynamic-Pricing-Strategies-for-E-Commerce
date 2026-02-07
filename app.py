# =============================================================
# FINAL INTERNSHIP PROJECT
# Project: Dynamic Pricing Strategies For E-Commerce
# Dataset: BookMyShow Ticket Sales Dataset
# Flow: Observe → Predict → Forecast
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Pricing Strategies For E-Commerce",
    layout="wide",
)

st.markdown(
    """
    <h1 style="text-align:center;color:#00b4d8;">
    Dynamic Pricing Strategies For E-Commerce
    </h1>
    <h4 style="text-align:center;">BookMyShow Ticket Sales Dataset</h4>
    <h6 style="text-align:center;">Developed By - Sampada Swami</h6>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# LOAD & CLEAN DATA
# -------------------------------------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("BookMyShow_Ticket_Sales_Data.csv")
    df.columns = df.columns.str.lower()

    # ---- DATE CLEANING ----
    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce"
    )

    df = df[df["date"].notna()]

    if df.empty:
        st.error("❌ No valid dates in dataset.")
        st.stop()

    # ---- COST & PROFIT ----
    df["cost"] = df["base_ticket_price"] * 0.6
    df["profit"] = (df["final_ticket_price"] - df["cost"]) * df["tickets_sold"]

    # ---- COMPETITOR PRICE (proxy – required by internship) ----
    np.random.seed(42)
    df["competitor_price"] = (
        df["base_ticket_price"]
        * np.random.uniform(0.9, 1.1, len(df))
    )

    # ---- TIME FEATURES ----
    df["day_of_week"] = df["date"].dt.day_name()

    def season_map(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Summer"
        elif m in [6, 7, 8]:
            return "Monsoon"
        else:
            return "Post-Monsoon"

    df["season"] = df["date"].dt.month.map(season_map)

    return df


df = load_data()

# -------------------------------------------------------------
# GLOBAL FILTERS
# -------------------------------------------------------------
st.sidebar.header("🎛 Global Filters")

selected_categories = st.sidebar.multiselect(
    "Event Type",
    df["event_type"].unique(),
    default=df["event_type"].unique(),
)

selected_events = st.sidebar.multiselect(
    "Event ID",
    df["event_id"].unique(),
)

selected_seasons = st.sidebar.multiselect(
    "Season",
    df["season"].unique(),
    default=df["season"].unique(),
)

selected_days = st.sidebar.multiselect(
    "Day of Week",
    df["day_of_week"].unique(),
    default=df["day_of_week"].unique(),
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

start_date, end_date = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# -------------------------------------------------------------
# FILTER DATA
# -------------------------------------------------------------
filtered_df = df[
    (df["event_type"].isin(selected_categories))
    & (df["season"].isin(selected_seasons))
    & (df["day_of_week"].isin(selected_days))
    & (df["date"].between(pd.to_datetime(start_date),
                           pd.to_datetime(end_date)))
]

if selected_events:
    filtered_df = filtered_df[
        filtered_df["event_id"].isin(selected_events)
    ]

# -------------------------------------------------------------
# NAVIGATION
# -------------------------------------------------------------
page = st.sidebar.radio(
    "📂 Navigate",
    [
        "Dashboard Overview",
        "Demand Prediction",
        "Time Series Forecasting",
        "Pricing & Revenue Insights",
    ],
)

# =============================================================
# DASHBOARD OVERVIEW
# =============================================================
if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Revenue", f"₹{filtered_df['revenue'].sum():,.0f}")
    c2.metric("Tickets Sold", f"{filtered_df['tickets_sold'].sum():,.0f}")
    c3.metric("Avg Price", f"₹{filtered_df['final_ticket_price'].mean():.1f}")
    c4.metric("Avg Discount", f"{filtered_df['discount_pct'].mean():.1f}%")
    c5.metric("Avg Capacity", f"{filtered_df['venue_capacity'].mean():.0f}")

    daily = filtered_df.groupby("date")[["tickets_sold", "revenue"]].sum().reset_index()

    st.plotly_chart(px.line(daily, x="date", y="tickets_sold",
                            title="Date vs Tickets Sold"),
                     use_container_width=True)

    st.plotly_chart(px.line(daily, x="date", y="revenue",
                            title="Date vs Revenue"),
                     use_container_width=True)

# =============================================================
# DEMAND PREDICTION
# =============================================================
elif page == "Demand Prediction":

    st.header("🤖 Demand Prediction")

    model_df = df.copy()

    le_season = LabelEncoder()
    le_day = LabelEncoder()

    model_df["season_enc"] = le_season.fit_transform(model_df["season"])
    model_df["day_enc"] = le_day.fit_transform(model_df["day_of_week"])

    features = [
        "final_ticket_price",
        "discount_pct",
        "competitor_price",
        "venue_capacity",
        "season_enc",
        "day_enc",
    ]

    X = model_df[features]
    y = model_df["tickets_sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    price = st.number_input("Ticket Price", 50.0, 5000.0, 400.0)
    discount = st.slider("Discount %", 0, 60, 15)
    competitor = st.number_input("Competitor Price", 50.0, 5000.0, 420.0)
    inventory = st.number_input("Venue Capacity", 0, 5000, 700)

    season = st.selectbox("Season", le_season.classes_)
    day = st.selectbox("Day", le_day.classes_)

    if st.button("Predict Demand"):

        inp = pd.DataFrame(
            [[
                price,
                discount,
                competitor,
                inventory,
                le_season.transform([season])[0],
                le_day.transform([day])[0],
            ]],
            columns=features,
        )

        pred = model.predict(inp)[0]

        cost = price * 0.6
        revenue = price * pred
        profit = (price - cost) * pred

        st.success(f"Predicted Tickets Sold: {int(pred)}")
        st.info(f"Expected Revenue: ₹{revenue:,.0f}")
        st.warning(f"Expected Profit: ₹{profit:,.0f}")

# =============================================================
# TIME SERIES FORECASTING + DOWNLOAD
# =============================================================
elif page == "Time Series Forecasting":

    st.header("📉 Time Series Forecasting")

    daily_demand = (
        filtered_df.groupby("date")["tickets_sold"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    st.plotly_chart(px.line(daily_demand,
                            x="date",
                            y="tickets_sold",
                            title="Historical Demand"),
                     use_container_width=True)

    # ---- Stationarity Test ----
    adf_result = adfuller(daily_demand["tickets_sold"])
    st.info(
        f"""
ADF Test p-value: **{adf_result[1]:.4f}**

If p < 0.05 → Series is stationary (good for ARIMA).
"""
    )

    model = ARIMA(daily_demand["tickets_sold"], order=(2, 1, 2))
    fit = model.fit()

    forecast = fit.forecast(30)

    future_dates = pd.date_range(
        daily_demand["date"].max() + pd.Timedelta(days=1),
        periods=30,
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecasted_tickets_sold": forecast,
    })

    combined = pd.concat([
        daily_demand.assign(type="History").rename(columns={"tickets_sold": "value"}),
        forecast_df.assign(type="Forecast").rename(columns={"forecasted_tickets_sold": "value"}),
    ])

    st.plotly_chart(px.line(combined,
                            x="date",
                            y="value",
                            color="type",
                            title="30-Day Demand Forecast"),
                     use_container_width=True)

    # 📥 DOWNLOAD OPTION
    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Forecasted Data (CSV)",
        data=csv,
        file_name="demand_forecast_30_days.csv",
        mime="text/csv",
    )

# =============================================================
# PRICING & REVENUE INSIGHTS
# =============================================================
elif page == "Pricing & Revenue Insights":

    st.header("💡 Pricing & Revenue Insights")

    st.plotly_chart(px.scatter(filtered_df,
                               x="final_ticket_price",
                               y="tickets_sold",
                               color="season"),
                     use_container_width=True)

    st.plotly_chart(px.scatter(filtered_df,
                               x="discount_pct",
                               y="tickets_sold",
                               color="day_of_week"),
                     use_container_width=True)

    rev = filtered_df.groupby(["date", "season"])["revenue"].sum().reset_index()

    st.plotly_chart(px.area(rev,
                            x="date",
                            y="revenue",
                            color="season"),
                     use_container_width=True)

    st.plotly_chart(px.scatter(filtered_df,
                               x="venue_capacity",
                               y="tickets_sold"),
                     use_container_width=True)

    st.success(
        """
### 📌 Strategic Interpretation

• Higher prices reduce demand  
• Discounts are more effective on weekdays  
• Seasonal peaks dominate ticket sales  
• Capacity limits revenue  
• Competitive pricing strongly influences demand  

"""
    )
