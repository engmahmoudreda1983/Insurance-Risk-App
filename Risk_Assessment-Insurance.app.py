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
st.markdown("Analyze customer risk profiles using advanced AI and explainable data metrics.")
st.markdown("---")

# --- 4. Sidebar (بيانات العميل) ---
st.sidebar.header("📝 Customer Input")
with st.sidebar.expander("👤 Personal & Financial", expanded=True):
    age = st.slider("Age", 18, 75, 35)
    income = st.number_input("Annual Income ($)", 30000, 300000, 70000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)

with st.sidebar.expander("🚗 History & Health", expanded=False):
    past_claims = st.number_input("Past Claims Count", 0, 10, 0)
    traffic_tickets = st.number_input("Traffic Tickets", 0, 10, 0)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")

# مدخلات إضافية مخفية للحفاظ على توافق الموديل (Defaults)
dependents, edu_level, savings, chronic, exercise, bp_sys, tenure, claims_amount, missed_payments, property_val, vehicle_age, commute_km = 1, 2, 20000, 0, 3, 120, 24, 0, 0, 200000, 5, 30

# --- 5. زر التحليل ---
if st.sidebar.button("🔍 Run Full Risk Analysis", use_container_width=True):
    # تجهيز الداتا
    input_data = pd.DataFrame([[age, dependents, edu_level, income, credit_score, debt_ratio, savings, bmi, smoking, chronic, exercise, bp_sys, tenure, past_claims, claims_amount, traffic_tickets, missed_payments, property_val, vehicle_age, (commute_km * 0.621)]], 
                               columns=['Age', 'Dependents', 'Education_Level', 'Annual_Income', 'Credit_Score', 'Debt_to_Income_Ratio', 'Savings_Amount', 'BMI', 'Smoking_Status', 'Chronic_Diseases', 'Exercise_Days_Per_Week', 'Blood_Pressure_Sys', 'Policy_Tenure_Months', 'Past_Claims_Count', 'Total_Claims_Amount', 'Traffic_Tickets', 'Missed_Payments', 'Property_Value', 'Vehicle_Age', 'Daily_Commute_Miles'])

    # التوقع
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    prob = xgb_model.predict_proba(input_pca)[0]
    res = le.inverse_transform([np.argmax(prob)])[0]
    
    # حساب سكور العداد
    risk_score = int((prob[0] * 10) + (prob[1] * 50) + (prob[2] * 90))

    # --- 6. العرض البصري ---
    st.subheader("📊 Assessment Results")
    col1, col2 = st.columns([1, 1])

    with col1:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_score,
            title={'text': f"Risk Status: {res}"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red" if risk_score > 66 else "orange" if risk_score > 33 else "green"}}
        ))
        st.plotly_chart(fig_g, use_container_width=True)

    with col2:
        # حساب تأثير العناصر (أساس المعادلة)
        st.markdown("### 🔑 Why this score?")
        impact_map = {
            "Credit History": (850 - credit_score) / 550,
            "Claims History": past_claims / 10,
            "Debt Stress": debt_ratio,
            "Driving Behavior": traffic_tickets / 10,
            "Health Risk (BMI/Smoking)": (bmi/50) + (0.3 if smoking else 0)
        }
        df_impact = pd.DataFrame(impact_map.items(), columns=['Factor', 'Weight']).sort_values('Weight', ascending=True)
        fig_bar = px.bar(df_impact, x='Weight', y='Factor', orientation='h', color='Weight', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- الجزء الجديد: شرح المعادلة (المساحة الفاضية) ---
    st.markdown("---")
    st.subheader("📝 Decision Logic & Factor Breakdown")
    st.write("Below is how each input contributed to the final AI decision logic:")
    
    breakdown_cols = st.columns(4)
    breakdown_cols[0].metric("Financial Impact", f"{int(debt_ratio*100)}%", delta="High Risk" if debt_ratio > 0.4 else "Stable")
    breakdown_cols[1].metric("Credit Reliability", f"{credit_score} pts", delta="- Risk" if credit_score > 700 else "+ Risk", delta_color="inverse")
    breakdown_cols[2].metric("Health Factor", f"BMI {bmi}", delta="Elevated" if bmi > 25 else "Normal")
    breakdown_cols[3].metric("AI Confidence", f"{max(prob)*100:.1f}%")

else:
    st.info("👈 Enter customer data in the sidebar and click 'Analyze' to fill this report.")