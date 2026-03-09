import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Smart Risk Assessor", page_icon="🛡️", layout="wide")

# --- 2. تحميل النماذج ---
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_model.pkl')
    le = joblib.load('label_encoder.pkl')
    xgb_model = joblib.load('xgboost_risk_model.pkl')
    return scaler, pca, le, xgb_model

scaler, pca, le, xgb_model = load_models()

# --- 3. تصميم الرأس ---
st.title("🛡️ Smart Insurance Risk Assessment")
st.markdown("Comprehensive AI Underwriting Dashboard - Detailed Risk Factor Analysis")
st.markdown("---")

# --- 4. الشريط الجانبي (كل التفاصيل القديمة) ---
st.sidebar.header("📝 Customer Information")

with st.sidebar.expander("👤 Personal & Health", expanded=True):
    age = st.slider("Age", 18, 75, 35)
    dependents = st.number_input("Dependents", 0, 5, 1)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")
    exercise = st.slider("Exercise Days/Week", 0, 7, 3)
    chronic = st.number_input("Chronic Diseases", 0, 5, 0)

with st.sidebar.expander("💰 Financial Data", expanded=False):
    income = st.number_input("Annual Income ($)", 30000, 300000, 70000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    savings = st.number_input("Savings Amount ($)", 0, 200000, 20000)

with st.sidebar.expander("🚗 Insurance History", expanded=False):
    past_claims = st.number_input("Past Claims Count", 0, 10, 0)
    traffic_tickets = st.number_input("Traffic Tickets", 0, 10, 0)
    tenure = st.slider("Policy Tenure (Months)", 0, 240, 24)
    missed_payments = st.number_input("Missed Payments", 0, 10, 0)

# قيم تقنية ثابتة للحفاظ على أبعاد الموديل
edu_level, bp_sys, claims_amount, property_val, vehicle_age, commute_km = 2, 120, 0, 200000, 5, 30

# --- 5. زر التوقع ومعالجة البيانات ---
if st.sidebar.button("🔍 Run Full Risk Analysis", use_container_width=True):
    with st.spinner('Calculating Assessment...'):
        
        # تجهيز الداتا للموديل
        input_data = pd.DataFrame([[age, dependents, edu_level, income, credit_score, debt_ratio, savings, bmi, smoking, chronic, exercise, bp_sys, tenure, past_claims, claims_amount, traffic_tickets, missed_payments, property_val, vehicle_age, (commute_km * 0.621)]], 
                                   columns=['Age', 'Dependents', 'Education_Level', 'Annual_Income', 'Credit_Score', 'Debt_to_Income_Ratio', 'Savings_Amount', 'BMI', 'Smoking_Status', 'Chronic_Diseases', 'Exercise_Days_Per_Week', 'Blood_Pressure_Sys', 'Policy_Tenure_Months', 'Past_Claims_Count', 'Total_Claims_Amount', 'Traffic_Tickets', 'Missed_Payments', 'Property_Value', 'Vehicle_Age', 'Daily_Commute_Miles'])

        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)
        prediction_proba = xgb_model.predict_proba(input_pca)[0] 
        result = le.inverse_transform([np.argmax(prediction_proba)])[0]
        confidence = np.max(prediction_proba) * 100

        # حساب سكور العداد الديناميكي
        gauge_val = int((prediction_proba[0] * 10) + (prediction_proba[1] * 50) + (prediction_proba[2] * 90))
        g_color = "#00CC96" if result == 'Low' else "#FECB52" if result == 'Medium' else "#EF553B"

        # --- 6. عرض الداش بورد المتكاملة ---
        st.subheader("📊 Visual Risk Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = gauge_val,
                title = {'text': f"Overall Risk: {result}", 'font': {'size': 24}},
                number = {'suffix': "%", 'font': {'color': g_color}},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': g_color}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            categories = ['Financial Stress', 'Driving Risk', 'Health (BMI)', 'Age Factor', 'Claims History']
            values = [min((debt_ratio/0.5)*100, 100), min((traffic_tickets/3)*100, 100), max(0, min(((bmi-18.5)/21.5)*100, 100)), max(0, min(((age-18)/57)*100, 100)), min((past_claims/3)*100, 100)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', line_color=g_color))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Risk Dimensions")
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- 7. رسمة الـ Bar Chart (تأثير كل عنصر) ---
        st.markdown("---")
        st.subheader("🔍 Feature Importance (How each factor affected the score)")
        
        impact_map = {
            "Credit History": (850 - credit_score) / 5.5,
            "Past Claims": past_claims * 10,
            "Debt-to-Income": debt_ratio * 100,
            "Traffic Violations": traffic_tickets * 15,
            "Health & Lifestyle": (bmi-18) + (25 if smoking else 0)
        }
        df_impact = pd.DataFrame(impact_map.items(), columns=['Factor', 'Impact Weight']).sort_values('Impact Weight', ascending=True)
        fig_bar = px.bar(df_impact, x='Impact Weight', y='Factor', orientation='h', color='Impact Weight', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- 8. الـ Key Points (Summary Metrics) ---
        st.markdown("---")
        st.subheader("📝 Key Decision Points")
        kp1, kp2, kp3, kp4 = st.columns(4)
        kp1.metric("Financial Stress", f"{int(debt_ratio*100)}%", "Critical" if debt_ratio > 0.4 else "Stable")
        kp2.metric("Credit Reliability", f"{credit_score}", "High Risk" if credit_score < 600 else "Trusted", delta_color="inverse")
        kp3.metric("Behavioral History", f"{traffic_tickets} Tickets", "Caution" if traffic_tickets > 2 else "Clean")
        kp4.metric("AI Confidence", f"{confidence:.1f}%")

else:
    st.info("👈 Enter details in the sidebar and click **Analyze** to generate the full dashboard.")