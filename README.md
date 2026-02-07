# Dynamic Pricing Strategies For E-Commerce

**BookMyShow Ticket Sales Dataset**

## 📌 Internship Final Project

This project is a **Streamlit-based analytics application**
demonstrating:

-   **Descriptive Analytics (EDA)**
-   **Predictive Modeling (Demand Prediction)**
-   **Time Series Forecasting (ARIMA)**

Analytical Flow:

> **Observe → Predict → Forecast**

------------------------------------------------------------------------

## 📂 Dataset Columns Used

The project is aligned to the actual dataset schema:

-   `date`
-   `event_id`
-   `event_type`
-   `city`
-   `language`
-   `venue_capacity`
-   `base_ticket_price`
-   `discount_pct`
-   `final_ticket_price`
-   `tickets_sold`
-   `occupancy_rate`
-   `revenue`

Additional engineered features:

-   `cost`
-   `profit`
-   `competitor_price` (proxy)
-   `season`
-   `day_of_week`

------------------------------------------------------------------------

## 🚀 Application Features

### ✅ 1. Dashboard Overview (EDA)

KPIs:

-   Total Revenue
-   Tickets Sold
-   Average Ticket Price
-   Average Discount %
-   Average Capacity

Visualizations:

-   Date vs Tickets Sold
-   Date vs Revenue
-   Season vs Demand
-   Day-of-Week Patterns
-   Heatmap (Day × Season)
-   Event Type vs Demand
-   Price vs Demand

------------------------------------------------------------------------

### ✅ 2. Demand Prediction

Machine Learning Model:

-   RandomForest Regressor

Input Drivers:

-   Ticket Price
-   Discount %
-   Competitor Price
-   Venue Capacity
-   Season
-   Day of Week

Outputs:

-   Predicted Tickets Sold
-   Expected Revenue
-   Expected Profit

------------------------------------------------------------------------

### ✅ 3. Time Series Forecasting

-   Daily demand aggregation
-   ADF stationarity test
-   ARIMA model
-   30-day demand forecast
-   Forecast download as CSV

------------------------------------------------------------------------

### ✅ 4. Pricing & Revenue Insights

-   Price vs Demand (by Season)
-   Discount Impact
-   Revenue Over Time
-   Capacity vs Sales
-   Strategic interpretations

------------------------------------------------------------------------

## 🎛 Global Filters

Available on all pages:

-   Event Type
-   Event ID
-   Season
-   Day of Week
-   Date Range

------------------------------------------------------------------------

## 🛠 Installation & Run

### 1️⃣ Create Environment

``` bash
conda create -n pricing python=3.10
conda activate pricing
```

### 2️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

Or:

``` bash
pip install streamlit pandas numpy plotly scikit-learn statsmodels
```

------------------------------------------------------------------------

### 3️⃣ Run App

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📁 Project Structure

    DynamicPricingProject/
    │
    ├── app.py
    ├── BookMyShow_Ticket_Sales_Data.csv
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

## 📊 Business Insights

-   Demand decreases with higher prices
-   Discounts are effective on weekdays
-   Seasonal peaks drive ticket sales
-   Inventory constraints limit revenue
-   Competitive pricing impacts customer behavior

------------------------------------------------------------------------

## 🎓 Internship Submission Checklist

✔ Dataset-aligned columns\
✔ Global filters applied\
✔ EDA dashboards\
✔ Demand prediction\
✔ ARIMA forecasting\
✔ Forecast download\
✔ Strategic interpretation

------------------------------------------------------------------------

## 👩‍💻 Developed By

**Sampada Swami**
