import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Smart Risk Assessor", page_icon="🛡️", layout="wide")

# --- 2. تحميل النماذج (باستخدام Cache لتسريع التطبيق) ---
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_model.pkl')
    le = joblib.load('label_encoder.pkl')
    xgb_model = joblib.load('xgboost_risk_model.pkl')
    return scaler, pca, le, xgb_model

scaler, pca, le, xgb_model = load_models()

# --- 3. تصميم رأس الصفحة (وصف إداري ومفهوم) ---
st.title("🛡️ Smart Insurance Risk Assessment")
st.markdown("""
Welcome to the **Comprehensive Customer Risk Profiler**. 
This smart application instantly analyzes a customer's personal, financial, health, and driving data 
to evaluate their overall insurance risk. It is designed to assist underwriters in making 
fast, accurate, and data-driven policy decisions.
""")
# خلينا خط فاصل واحد بس هنا
st.markdown("---")

# --- 4. الشريط الجانبي (Sidebar) لإدخال بيانات العميل ---
st.sidebar.header("📝 Customer Information")

with st.sidebar.expander("👤 Personal Details", expanded=True):
    age = st.sidebar.slider("Age", 18, 75, 35)
    dependents = st.sidebar.number_input("Dependents (Children/Spouse)", 0, 5, 1)
    
    edu_mapping = {
        1: "1 - High School / No Degree", 
        2: "2 - Bachelor's Degree", 
        3: "3 - Master's Degree", 
        4: "4 - Doctorate / PhD"
    }
    edu_level = st.sidebar.selectbox("Education Level", options=[1, 2, 3, 4], format_func=lambda x: edu_mapping[x], help="Select the highest academic degree completed by the customer.")

with st.sidebar.expander("💰 Financial Data", expanded=False):
    income = st.number_input("Annual Income ($)", 30000, 300000, 70000)
    credit_score = st.slider("Credit Score", 300, 850, 650, help="Standard FICO score ranging from 300 (Poor) to 850 (Excellent).")
    debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, help="Percentage of monthly gross income that goes toward paying debts (e.g., 0.30 = 30%).")
    savings = st.number_input("Savings Amount ($)", 0, 200000, 20000)

with st.sidebar.expander("❤️ Health & Lifestyle", expanded=False):
    bmi = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0, help="Underweight < 18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese > 30.")
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")
    chronic = st.number_input("Chronic Diseases", 0, 5, 0, help="Number of chronic conditions (e.g., Diabetes, Hypertension, Asthma).")
    exercise = st.slider("Exercise Days/Week", 0, 7, 3)
    
    st.write("Blood Pressure")
    col_bp1, col_bp2 = st.columns(2)
    with col_bp1:
        bp_sys = st.number_input("Systolic (Top)", 90, 200, 120, help="Normal is around 120.")
    with col_bp2:
        bp_dia = st.number_input("Diastolic (Bottom)", 60, 130, 80, help="Normal is around 80.")

with st.sidebar.expander("📄 Insurance & Asset History", expanded=False):
    tenure = st.slider("Policy Tenure (Months)", 0, 240, 24, help="How long the customer has been insured with the company.")
    past_claims = st.number_input("Past Claims Count", 0, 10, 0, help="Number of insurance claims filed in the past.")
    claims_amount = st.number_input("Total Claims Amount ($)", 0, 50000, 0, help="Total monetary value of all past claims.")
    traffic_tickets = st.number_input("Traffic Tickets", 0, 10, 0, help="Number of traffic violations in the last 3 years.")
    missed_payments = st.number_input("Missed Payments", 0, 10, 0, help="Number of times the customer missed a premium payment.")
    property_val = st.number_input("Property Value ($)", 50000, 1000000, 200000)
    vehicle_age = st.slider("Vehicle Age (Years)", 0, 30, 5)
    commute_km = st.slider("Daily Commute (km)", 0, 250, 30, help="Average kilometers driven per day by the customer.")

# --- 5. زر التوقع ومعالجة البيانات ---
if st.sidebar.button("🔍 Analyze Risk Profile", use_container_width=True):
    with st.spinner('Calculating risk profile...'):
        
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

        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)
        prediction_encoded = xgb_model.predict(input_pca)
        prediction_proba = xgb_model.predict_proba(input_pca)
        
        result = le.inverse_transform(prediction_encoded)[0]
        confidence = np.max(prediction_proba) * 100

        # --- 6. عرض النتيجة والرسومات للمستخدم ---
        # شيلنا الخط الفاصل التاني من هنا عشان ميعملش الفراغ
        st.subheader("📊 Visual Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        if result == 'Low':
            gauge_val = 15
            gauge_color = "#00CC96" 
            status_text = "Safe"
            st.success(f"### 🟢 Customer is safe to insure.")
        elif result == 'Medium':
            gauge_val = 50
            gauge_color = "#FECB52" 
            status_text = "Warning"
            st.warning(f"### 🟡 Proceed with standard checks.")
        else:
            gauge_val = 85
            gauge_color = "#EF553B" 
            status_text = "Risky"
            st.error(f"### 🔴 Requires manual underwriter review.")

        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = gauge_val,
                title = {'text': f"Overall Risk: {status_text}", 'font': {'size': 24}},
                number = {'font': {'color': gauge_color}, 'suffix': "%"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 33], 'color': "rgba(0, 204, 150, 0.2)"},
                        {'range': [33, 66], 'color': "rgba(254, 203, 82, 0.2)"},
                        {'range': [66, 100], 'color': "rgba(239, 85, 59, 0.2)"}],
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            categories = ['Financial Stress', 'Driving Risk', 'Health (BMI)', 'Age Factor', 'Claims History']
            values = [
                min(debt_ratio * 100, 100),               
                min((traffic_tickets / 5) * 100, 100),    
                min((bmi / 40) * 100, 100),               
                min((age / 75) * 100, 100),               
                min((past_claims / 5) * 100, 100)         
            ]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
              r=values + [values[0]], 
              theta=categories + [categories[0]],
              fill='toself',
              line_color=gauge_color
            ))
            fig_radar.update_layout(
              polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
              showlegend=False,
              height=350, margin=dict(l=40, r=40, t=50, b=20),
              title={'text': "Customer Risk Breakdown", 'font': {'size': 20}}
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        st.info(f"🎯 **AI Confidence Level:** {confidence:.2f}%")
else:
    # الرسالة التوجيهية بتظهر لو الموظف لسه مداسش على الزرار
    st.info("👈 Please enter the customer's details in the sidebar and click **Analyze Risk Profile** to generate the assessment.")