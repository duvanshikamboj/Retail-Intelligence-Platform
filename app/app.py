import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf(
    category,
    sub_category,
    market,
    segment,
    ship_mode,
    quantity,
    discount,
    predicted_sales,
    metrics
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>AI Retail Intelligence</b>", styles["Title"]))

    story.append(Paragraph("<br/>Sales Prediction Report<br/><br/>", styles["Heading2"]))

    story.append(Paragraph(f"<b>Category:</b> {category}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Sub Category:</b> {sub_category}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Market:</b> {market}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Segment:</b> {segment}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Ship Mode:</b> {ship_mode}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Quantity:</b> {quantity}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Discount:</b> {discount}", styles["BodyText"]))

    story.append(Paragraph("<br/>", styles["BodyText"]))

    story.append(
        Paragraph(
            f"<b>Predicted Sales:</b> ${predicted_sales:.2f}",
            styles["Heading2"]
        )
    )

    story.append(Paragraph("<br/>", styles["BodyText"]))

    story.append(
        Paragraph(
            f"<b>Average CV R²:</b> {metrics['CV Mean R2']:.4f}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>MAE:</b> ${metrics['MAE']:.2f}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>RMSE:</b> ${metrics['RMSE']:.2f}",
            styles["BodyText"]
        )
    )

    doc.build(story)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Retail Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main{
    background-color:#f6f8fc;
}

.block-container{
    padding-top:1.5rem;
}

div[data-testid="stMetric"]{

    background:#1E293B;

    color:white;

    border-radius:15px;

    padding:18px;

    border-left:6px solid #3B82F6;

    box-shadow:0px 4px 12px rgba(0,0,0,0.3);
}

div[data-testid="stMetric"] label{

    color:white !important;

}

div[data-testid="stMetricValue"]{

    color:white !important;

}

div[data-testid="stMetricDelta"]{

    color:#22C55E !important;

}

h1{

    color:#2563EB;

    font-weight:bold;
}

h2{

    color:#0F172A;
}

h3{

    color:#334155;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD FILES
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load Trained XGBoost Pipeline
model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "xgboost_sales_model.pkl"
    )
)

# Load Dataset
df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "featured_superstore.csv"
    )
)

# Standardize Column Names
df.columns = (
    df.columns
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# Load Model Metrics
metrics = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "model_metrics.pkl"
    )
)

# =====================================================
# LOAD FEATURE IMPORTANCE
# =====================================================

feature_importance = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "feature_importance.pkl"
    )
)

prediction_results = joblib.load(

    os.path.join(

        BASE_DIR,

        "models",

        "prediction_results.pkl"

    )

)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📊 AI Retail Intelligence")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "🤖 Sales Prediction",

        "📈 Dashboard",

        "📊 Feature Importance",

        "📈 Model Performance",

        "💡 Business Insights",

        "ℹ About Project"

    ]

)

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("📊 AI Retail Intelligence & Demand Forecasting Platform") 

    st.markdown("")

    st.write("""

Welcome to the AI Retail Intelligence Platform.

This project predicts future retail sales using Machine Learning and Advanced Feature Engineering.

The prediction model is built using an optimized XGBoost Regressor.

The application helps businesses estimate future sales, analyze trends, monitor KPIs, and generate business insights for better decision making.

""")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Dataset Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Features",
            f"{df.shape[1]}"
        )

    with col3:

        st.metric(
            "Algorithm",
            "XGBoost"
        )

    st.markdown("---")

    st.subheader("Project Workflow")

    st.success("✔ Data Collection")

    st.success("✔ Data Cleaning")

    st.success("✔ Feature Engineering")

    st.success("✔ Exploratory Data Analysis")

    st.success("✔ Machine Learning")

    st.success("✔ Sales Prediction")

    st.success("✔ Business Insights")

    st.success("✔ Interactive Dashboard")

    # =====================================================
# SALES PREDICTION
# =====================================================

elif page == "🤖 Sales Prediction":

    st.title("🤖 AI Sales Prediction")

    st.markdown("---")

    st.subheader("Enter Product Details")

    col1, col2 = st.columns(2)

    # ==========================
    # LEFT COLUMN
    # ==========================

    with col1:

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=2
        )

        discount = st.slider(
            "Discount",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

        shipping_cost = st.number_input(
            "Shipping Cost",
            min_value=0.0,
            value=20.0
        )

        year = st.selectbox(
            "Order Year",
            sorted(df["order_year"].unique())
        )

        month = st.selectbox(
            "Order Month",
            sorted(df["order_month"].unique())
        )

        quarter = st.selectbox(
            "Order Quarter",
            sorted(df["order_quarter"].unique())
        )

    # ==========================
    # RIGHT COLUMN
    # ==========================

    with col2:

        weekend = st.selectbox(
            "Weekend",
            [
                "No",
                "Yes"
            ]
        )

        category = st.selectbox(
            "Category",
            sorted(df["category"].dropna().unique())
        )

        sub_category = st.selectbox(
            "Sub Category",
            sorted(df["sub_category"].dropna().unique())
        )

        market = st.selectbox(
            "Market",
            sorted(df["market"].dropna().unique())
        )

        segment = st.selectbox(
            "Segment",
            sorted(df["segment"].dropna().unique())
        )

        ship_mode = st.selectbox(
            "Ship Mode",
            sorted(df["ship_mode"].dropna().unique())
        )

    st.markdown("---")

    predict_button = st.button(
        "🚀 Predict Sales",
        use_container_width=True
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_button:

        expected_price = (
            df[
                df["sub_category"] == sub_category
            ]["sales"].median()
        )

        if pd.isna(expected_price):

            expected_price = df["sales"].median()

        input_data = {

            "quantity": quantity,

            "discount": discount,

            "subcat_expected_price": expected_price,

            "order_month": month,

            "order_year": year,

            "order_quarter": quarter,

            "is_weekend": 1 if weekend == "Yes" else 0,

            "category": category,

            "sub_category": sub_category,

            "market": market,

            "segment": segment,

            "ship_mode": ship_mode

        }

        if "shipping_cost" in df.columns:

            input_data["shipping_cost"] = shipping_cost

        if "profit" in df.columns:

            input_data["profit"] = 0

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)

        predicted_sales = float(prediction[0])

        st.markdown("---")

        st.success("Prediction Completed Successfully!")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Predicted Sales",

                f"${predicted_sales:,.2f}"

            )

        with col2:

            if predicted_sales < 100:

                st.error("Low Expected Sales")

            elif predicted_sales < 500:

                st.warning("Medium Expected Sales")

            else:

                st.success("High Expected Sales")

        pdf = create_pdf(
            category,
            sub_category,
            market,
            segment,
            ship_mode,
            quantity,
            discount,
            predicted_sales,
            metrics
        )

        st.download_button(

            label="📄 Download PDF Report",

            data=pdf,

            file_name="Sales_Prediction_Report.pdf",

            mime="application/pdf"

        )

        st.markdown("---")

        st.subheader("Input Summary")

        st.dataframe(

            input_df,

            use_container_width=True

        )
        # =====================================================
# DASHBOARD
# =====================================================

elif page == "📈 Dashboard":

    st.title("📈 AI Retail Dashboard")

    st.markdown("---")

    # =====================================================
    # KPI CARDS
    # =====================================================

    total_sales = df["sales"].sum()

    total_profit = (
        df["profit"].sum()
        if "profit" in df.columns
        else 0
    )

    total_orders = len(df)

    avg_sales = df["sales"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "💰 Total Sales",
            f"${total_sales:,.2f}"
        )

    with col2:

        st.metric(
            "📈 Total Profit",
            f"${total_profit:,.2f}"
        )

    with col3:

        st.metric(
            "🛒 Total Orders",
            f"{total_orders:,}"
        )

    with col4:

        st.metric(
            "💵 Avg Sales",
            f"${avg_sales:,.2f}"
        )

    st.markdown("---")

    # =====================================================
    # SALES BY CATEGORY
    # =====================================================

    st.subheader("📦 Sales by Category")

    category_sales = (

        df.groupby("category")["sales"]

        .sum()

        .reset_index()

    )

    fig = px.bar(

        category_sales,

        x="category",

        y="sales",

        color="category",

        text_auto=".2s",

        title="Category Wise Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # =====================================================
    # SALES BY MARKET
    # =====================================================

    st.subheader("🌍 Sales by Market")

    market_sales = (

        df.groupby("market")["sales"]

        .sum()

        .reset_index()

    )

    fig = px.pie(

        market_sales,

        names="market",

        values="sales",

        hole=0.5

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # =====================================================
    # MONTHLY SALES TREND
    # =====================================================

    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = (

        df.groupby("order_month")["sales"]

        .sum()

        .reset_index()

        .sort_values("order_month")

    )

    fig = px.line(

        monthly_sales,

        x="order_month",

        y="sales",

        markers=True,

        title="Monthly Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # =====================================================
    # TOP SUB-CATEGORIES
    # =====================================================

    st.subheader("🏆 Top 10 Sub Categories")

    top_products = (

        df.groupby("sub_category")["sales"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

        .reset_index()

    )

    fig = px.bar(

        top_products,

        x="sales",

        y="sub_category",

        orientation="h",

        color="sales",

        title="Top Selling Sub Categories"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # =====================================================
    # SALES VS PROFIT
    # =====================================================

    if "profit" in df.columns:

        st.subheader("📊 Sales vs Profit")

        fig = px.scatter(

            df,

            x="sales",

            y="profit",

            color="category",

            hover_data=["sub_category"],

            title="Sales vs Profit"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        # =====================================================
# FEATURE IMPORTANCE
# =====================================================

elif page == "📊 Feature Importance":

    st.title("📊 Feature Importance Analysis")

    st.markdown(
        """
        This chart shows which features have the greatest impact on
        the XGBoost model's sales predictions.
        """
    )

    st.markdown("---")

    # Top 15 Features
    top_features = feature_importance.head(15).copy()

    # Clean feature names
    top_features["Feature"] = (
        top_features["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )

    fig = px.bar(

        top_features,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        text_auto=".3f",

        title="Top 15 Most Important Features"

    )

    fig.update_layout(

        yaxis=dict(autorange="reversed"),

        height=650

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Feature Importance Table")

    st.dataframe(
        top_features,
        use_container_width=True
    )

# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.markdown("---")

    st.subheader("Actual vs Predicted Sales")

    fig = px.scatter(

        prediction_results,

        x="Actual Sales",

        y="Predicted Sales",

        opacity=0.6,

        title="Actual vs Predicted"

    )

    fig.add_shape(

        type="line",

        x0=prediction_results["Actual Sales"].min(),

        y0=prediction_results["Actual Sales"].min(),

        x1=prediction_results["Actual Sales"].max(),

        y1=prediction_results["Actual Sales"].max(),

        line=dict(

            color="red",

            width=2

        )

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Prediction Error Distribution")

    prediction_results["Residual"] = (

        prediction_results["Actual Sales"]

        -

        prediction_results["Predicted Sales"]

    )

    fig = px.histogram(

        prediction_results,

        x="Residual",

        nbins=40,

        title="Residual Distribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "R² Score",

            f"{metrics['R2 Score']:.4f}"

        )

    with col2:

        st.metric(

            "MAE",

            f"${metrics['MAE']:.2f}"

        )

    with col3:

        st.metric(

            "RMSE",

            f"${metrics['RMSE']:.2f}"

        )

    

        # =====================================================
# BUSINESS INSIGHTS
# =====================================================

elif page == "💡 Business Insights":

    st.title("💡 Business Insights")

    st.markdown("---")

    # =====================================================
    # BEST CATEGORY
    # =====================================================

    best_category = (
        df.groupby("category")["sales"]
        .sum()
        .idxmax()
    )

    st.success(
        f"🏆 Best Performing Category : {best_category}"
    )

    # =====================================================
    # BEST MARKET
    # =====================================================

    best_market = (
        df.groupby("market")["sales"]
        .sum()
        .idxmax()
    )

    st.info(
        f"🌍 Highest Revenue Market : {best_market}"
    )

    # =====================================================
    # MOST PROFITABLE SUB CATEGORY
    # =====================================================

    if "profit" in df.columns:

        best_subcategory = (
            df.groupby("sub_category")["profit"]
            .sum()
            .idxmax()
        )

        st.success(
            f"💰 Most Profitable Sub Category : {best_subcategory}"
        )

    # =====================================================
    # AVERAGE DISCOUNT
    # =====================================================

    avg_discount = df["discount"].mean()

    st.warning(
        f"🎯 Average Discount Offered : {avg_discount:.2%}"
    )

    # =====================================================
    # SALES DISTRIBUTION
    # =====================================================

    st.subheader("Sales Distribution")

    fig = px.histogram(

        df,

        x="sales",

        nbins=50,

        title="Distribution of Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # =====================================================
    # SALES VS DISCOUNT
    # =====================================================

    st.subheader("Sales vs Discount")

    fig = px.scatter(

        df,

        x="discount",

        y="sales",

        color="category",

        title="Impact of Discount on Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # =====================================================
    # INSIGHTS
    # =====================================================

    st.markdown("---")

    st.subheader("📌 Business Insights")

    st.success(
        "✔ Focus inventory on the highest selling categories."
    )

    st.success(
        "✔ Increase stock availability in high-performing markets."
    )

    st.success(
        "✔ Optimize discount strategies to improve profitability."
    )

    st.success(
        "✔ Promote top-performing sub-categories."
    )

    st.success(
        "✔ Monitor seasonal demand for better forecasting."
    )

# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "ℹ About Project":

    st.title("ℹ AI Retail Intelligence Platform")

    st.markdown("---")

    st.subheader("📌 Project Objective")

    st.write(
        """
This project predicts retail sales using Machine Learning.

The objective is to help businesses improve demand forecasting,
inventory planning and decision making.
"""
    )

    st.markdown("---")

    st.subheader("📊 Dataset")

    st.write(
        """
Global Superstore Dataset

• 51,000+ Orders

• Multiple Markets

• Multiple Categories

• Sales, Profit, Shipping Cost and Customer Information
"""
    )

    st.markdown("---")

    st.subheader("🤖 Machine Learning Model")

    st.write(
        """
Model Used :

✅ XGBoost Regressor

The model is trained using advanced feature engineering,
categorical encoding and preprocessing pipeline.
"""
    )

    st.markdown("---")

    st.subheader("🛠 Technologies Used")

    st.write(
        """
• Python

• Pandas

• NumPy

• Scikit-Learn

• XGBoost

• Streamlit

• Plotly

• Joblib
"""
    )

    st.markdown("---")

    st.subheader("📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "R² Score",
            f"{metrics['R2 Score']:.4f}"
        )

    with col2:

        st.metric(
            "MAE",
            f"${metrics['MAE']:.2f}"
        )

    with col3:

        st.metric(
            "RMSE",
            f"${metrics['RMSE']:.2f}"
        )

    st.markdown("---")

    st.subheader("🚀 Project Workflow")

    st.info("1️⃣ Data Collection")

    st.info("2️⃣ Data Cleaning")

    st.info("3️⃣ Feature Engineering")

    st.info("4️⃣ Exploratory Data Analysis")

    st.info("5️⃣ Model Training (XGBoost)")

    st.info("6️⃣ Sales Prediction")

    st.info("7️⃣ Business Insights")

    st.markdown("---")

    st.success(
        "Developed as an AI-powered Retail Intelligence & Demand Forecasting Project using Machine Learning and Streamlit."
    )