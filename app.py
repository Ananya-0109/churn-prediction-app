import streamlit as st
import joblib
import pandas as pd

# Load saved artifacts
model = joblib.load("xgb_churn_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction")
st.title("📉 Customer Churn Prediction App")

st.write("Fill customer details to predict churn probability")

# -------- USER INPUTS (RAW FEATURES) --------
total_spend = st.number_input("Total Spend", min_value=0.0)
orders_last_3_months = st.number_input("Orders in Last 3 Months", min_value=0)
cancelled_orders = st.number_input("Cancelled Orders", min_value=0)
delayed_orders = st.number_input("Delayed Orders", min_value=0)
refund_events = st.number_input("Refund Events", min_value=0)
is_subscribed = st.selectbox("Is Subscribed?", [0, 1])

# -------- FEATURE ENGINEERING (SAME AS TRAINING) --------
delivery_issues = cancelled_orders + delayed_orders + refund_events

high_value_users = int(total_spend > 4500)  # median-based approx
frequent_users = int(orders_last_3_months > 5)

# -------- CREATE INPUT DICTIONARY --------
input_dict = {
    "total_spend": total_spend,
    "orders_last_3_months": orders_last_3_months,
    "cancelled_orders": cancelled_orders,
    "delayed_orders": delayed_orders,
    "refund_events": refund_events,
    "delivery_issues": delivery_issues,
    "is_subscribed": is_subscribed,
    "high_value_users": high_value_users,
    "frequent_users": frequent_users,
}

# -------- PREDICTION --------
if st.button("Predict Churn"):
    input_df = pd.DataFrame([input_dict])

    # Align with training columns (CRITICAL)
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write(f"### 🔍 Churn Probability: **{probability:.2%}**")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")
