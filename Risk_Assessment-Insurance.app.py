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

# --- 3. تصميم رأس الصفحة ---
st.title("🛡️ Smart Insurance Risk Assessment")
st.markdown("""
Welcome to the **Comprehensive Customer Risk Profiler**. 
This smart application instantly analyzes a customer's personal, financial, health, and driving data 
to evaluate their overall insurance risk.
""")
st.markdown("---")

# --- 4. الشريط الجانبي (Sidebar) - نفس تفاصيل ملفك بالظبط ---
st.sidebar.header("📝 Customer Information")

with st.sidebar.expander("👤 Personal Details", expanded=True):
    age = st.sidebar.slider("Age", 18, 75, 35, help="Customer's age in years.")
    dependents = st.sidebar.number_input("Dependents (Children/Spouse)", 0, 5, 1)
    edu_mapping = {1: "1 - High School", 2: "2 - Bachelor", 3: "3 - Master", 4: "4 - Doctorate"}
    edu_level = st.sidebar.selectbox("Education Level", options=[1, 2, 3, 4], format_func=lambda x: edu_mapping[x])

with st.sidebar.expander("💰 Financial Data", expanded=False):
    income = st.number_input("Annual Income ($)", 30000, 300000, 70000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    savings = st.number_input("Savings Amount ($)", 0, 200000, 20000)

with st.sidebar.expander("❤️ Health & Lifestyle", expanded=False):
    bmi = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")
    chronic = st.number_input("Chronic Diseases", 0, 5, 0)
    exercise = st.slider("Exercise Days/Week", 0, 7, 3)
    st.write("Blood Pressure")
    col_bp1, col_bp2 = st.columns(2)
    with col_bp1: bp_sys = st.number_input("Systolic", 90, 200, 120)
    with col_bp2: bp_dia = st.number_input("Diastolic", 60, 130, 80)

with st.sidebar.expander("📄 Insurance & Asset History", expanded=False):
    tenure = st.slider("Policy Tenure (Months)", 0, 240, 24)
    past_claims = st.number_input("Past Claims Count", 0, 10, 0)
    claims_amount = st.number_input("Total Claims Amount ($)", 0, 50000, 0)
    traffic_tickets = st.number_input("Traffic Tickets", 0, 10, 0)
    missed_payments = st.number_input("Missed Payments", 0, 10, 0)
    property_val = st.number_input("Property Value ($)", 50000, 1000000, 200000)
    vehicle_age = st.slider("Vehicle Age (Years)", 0, 30, 5)
    commute_km = st.slider("Daily Commute (km)", 0, 250, 30)

# --- 5. زر التوقع ومعالجة البيانات ---
if st.sidebar.button("🔍 Analyze Risk Profile", use_container_width=True):
    with st.spinner('Calculating Assessment...'):
        
        commute_miles = commute_km * 0.621371
        input_data = pd.DataFrame({
            'Age': [age], 'Dependents': [dependents], 'Education_Level': [edu_level],
            'Annual_Income': [income], 'Credit_Score': [credit_score], 'Debt_to_Income_Ratio': [debt_ratio],
            'Savings_Amount': [savings], 'BMI': [bmi], 'Smoking_Status': [smoking],
            'Chronic_Diseases': [chronic], 'Exercise_Days_Per_Week': [exercise],
            'Blood_Pressure_Sys': [bp_sys], 'Policy_Tenure_Months': [tenure],
            'Past_Claims_Count': [past_claims], 'Total_Claims_Amount': [claims_amount],
            'Traffic_Tickets': [traffic_tickets], 'Missed_Payments': [missed_payments],
            'Property_Value': [property_val], 'Vehicle_Age': [vehicle_age], 'Daily_Commute_Miles': [commute_miles]
        })

        # Processing
        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)
        prediction_proba = xgb_model.predict_proba(input_pca)[0] 
        result = le.inverse_transform([np.argmax(prediction_proba)])[0]
        confidence = np.max(prediction_proba) * 100

        # حساب المؤشر الديناميكي
        gauge_val = int((prediction_proba[0] * 10) + (prediction_proba[1] * 50) + (prediction_proba[2] * 90))
        g_color = "#00CC96" if result == 'Low' else "#FECB52" if result == 'Medium' else "#EF553B"

        # --- 6. عرض النتائج والرسومات (العداد والعنكبوت) ---
        st.subheader("📊 Visual Risk Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = gauge_val,
                title = {'text': f"Overall Risk: {result}", 'font': {'size': 24}},
                number = {'font': {'color': g_color}, 'suffix': "%"},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': g_color}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            categories = ['Financial Stress', 'Driving Risk', 'Health (BMI)', 'Age Factor', 'Claims History']
            values = [min((debt_ratio/0.5)*100, 100), min((traffic_tickets/3)*100, 100), max(0, min(((bmi-18.5)/21.5)*100, 100)), max(0, min(((age-18)/57)*100, 100)), min((past_claims/3)*100, 100)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', line_color=g_color))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Risk Dimensions")
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- 7. إضافة الـ Bar Chart (تأثير العناصر - إجابة سؤالك: كل عنصر أثر بقد إيه) ---
        st.markdown("---")
        st.subheader("🔍 Risk Driver Analysis (Factor Impact)")
        
        impact_map = {
            "Credit Score Status": (850 - credit_score) / 5.5,
            "Claims History": past_claims * 10,
            "Debt Exposure": debt_ratio * 100,
            "Driving Violations": traffic_tickets * 15,
            "Lifestyle Factors": (bmi-18) + (25 if smoking else 0)
        }
        df_impact = pd.DataFrame(impact_map.items(), columns=['Factor', 'Impact Weight']).sort_values('Impact Weight', ascending=True)
        fig_bar = px.bar(df_impact, x='Impact Weight', y='Factor', orientation='h', color='Impact Weight', color_continuous_scale='Reds', title="How much each factor pushed the score up")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- 8. إضافة الـ Key Points (Summary Metrics) ---
        st.markdown("---")
        st.subheader("📝 Key Decision Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Financial Health", f"{int(debt_ratio*100)}%", "Critical" if debt_ratio > 0.4 else "Stable")
        k2.metric("Credit Grade", f"{credit_score}", "Action Needed" if credit_score < 600 else "High Quality", delta_color="inverse")
        k3.metric("Behavioral Risk", f"{traffic_tickets} Tickets", "Alert" if traffic_tickets > 2 else "Clean History")
        k4.metric("AI Certainty", f"{confidence:.1f}%")

else:
    st.info("👈 Enter customer details in the sidebar and click **Analyze Risk Profile** to generate the assessment.")