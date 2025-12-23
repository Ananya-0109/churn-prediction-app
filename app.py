if st.button("Predict Churn"):

    # ---- VALIDATION ----
    if cancelled_orders + delayed_orders + refund_events > orders_last_3_months:
        st.error("❌ Issues cannot exceed total orders")
        st.stop()

    delivery_issues = min(
        cancelled_orders + delayed_orders + refund_events,
        orders_last_3_months
    )

    high_value_users = int(total_spend >= 4500)
    frequent_users = int(orders_last_3_months >= 3)

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

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    probability = model.predict_proba(input_df)[0][1]

    THRESHOLD = 0.3
    prediction = int(probability >= THRESHOLD)

    st.write(f"### 🔍 Churn Probability: **{probability:.2%}**")

    if prediction:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")
