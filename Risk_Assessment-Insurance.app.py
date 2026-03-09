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
st.markdown("Comprehensive Underwriting Dashboard: Multi-Factor AI Risk Analysis")
st.markdown("---")

# --- 4. الشريط الجانبي (التفاصيل الكاملة) ---
st.sidebar.header("📝 Customer Information")

with st.sidebar.expander("👤 Personal Details", expanded=True):
    age = st.sidebar.slider("Age", 18, 75, 35)
    dependents = st.sidebar.number_input("Dependents", 0, 5, 1)
    edu_mapping = {1: "1 - High School", 2: "2 - Bachelor", 3: "3 - Master", 4: "4 - Doctorate"}
    edu_level = st.sidebar.selectbox("Education Level", options=[1, 2, 3, 4], format_func=lambda x: edu_mapping[x])

with st.sidebar.expander("💰 Financial Data", expanded=False):
    income = st.number_input("Annual Income ($)", 30000, 300000, 70000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    savings = st.number_input("Savings Amount ($)", 0, 200000, 20000)

with st.sidebar.expander("❤️ Health & Lifestyle", expanded=False):
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")
    chronic = st.number_input("Chronic Diseases", 0, 5, 0)
    exercise = st.slider("Exercise Days/Week", 0, 7, 3)
    st.write("Blood Pressure")
    col_bp1, col_bp2 = st.columns(2)
    with col_bp1: bp_sys = st.number_input("Systolic", 90, 200, 120)
    with col_bp2: bp_dia = st.number_input("Diastolic", 60, 130, 80)

with st.sidebar.expander("📄 Insurance History", expanded=False):
    tenure = st.slider("Policy Tenure (Months)", 0, 240, 24)
    past_claims = st.number_input("Past Claims Count", 0, 10, 0)
    traffic_tickets = st.number_input("Traffic Tickets", 0, 10, 0)
    missed_payments = st.number_input("Missed Payments", 0, 10, 0)

# قيم افتراضية للحفاظ على استقرار الموديل
claims_amount, property_val, vehicle_age, commute_km = 0, 200000, 5, 30

# --- 5. زر التوقع ومعالجة البيانات ---
if st.sidebar.button("🔍 Analyze Risk Profile", use_container_width=True):
    with st.spinner('Calculating Detailed Risk Profile...'):
        
        input_data = pd.DataFrame({
            'Age': [age], 'Dependents': [dependents], 'Education_Level': [edu_level],
            'Annual_Income': [income], 'Credit_Score': [credit_score], 'Debt_to_Income_Ratio': [debt_ratio],
            'Savings_Amount': [savings], 'BMI': [bmi], 'Smoking_Status': [smoking],
            'Chronic_Diseases': [chronic], 'Exercise_Days_Per_Week': [exercise],
            'Blood_Pressure_Sys': [bp_sys], 'Policy_Tenure_Months': [tenure],
            'Past_Claims_Count': [past_claims], 'Total_Claims_Amount': [claims_amount],
            'Traffic_Tickets': [traffic_tickets], 'Missed_Payments': [missed_payments],
            'Property_Value': [property_val], 'Vehicle_Age': [vehicle_age], 'Daily_Commute_Miles': [commute_km * 0.621]
        })

        # Processing
        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)
        prediction_proba = xgb_model.predict_proba(input_pca)[0] 
        result = le.inverse_transform([np.argmax(prediction_proba)])[0]
        confidence = np.max(prediction_proba) * 100

        # ⭐ تصحيح معادلة العداد (الربط بالأسماء لضمان الدقة)
        classes = le.classes_
        proba_dict = dict(zip(classes, prediction_proba))
        
        # الحسبة الصحيحة: Low ياخد سكور قليل، و High ياخد سكور عالي
        dynamic_score = (proba_dict.get('Low', 0) * 15) + (proba_dict.get('Medium', 0) * 50) + (proba_dict.get('High', 0) * 90)
        gauge_val = int(dynamic_score)
        
        g_color = "#00CC96" if result == 'Low' else "#FECB52" if result == 'Medium' else "#EF553B"

        # --- 6. العرض البصري (العداد والعنكبوت) ---
        st.subheader("📊 Visual Assessment Summary")
        c1, c2 = st.columns(2)
        
        with c1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = gauge_val,
                title = {'text': f"Final Risk: {result}", 'font': {'size': 24}},
                number = {'font': {'color': g_color}, 'suffix': "%"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': g_color},
                         'steps': [{'range': [0, 33], 'color': "#00CC9633"},
                                   {'range': [33, 66], 'color': "#FECB5233"},
                                   {'range': [66, 100], 'color': "#EF553B33"}]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c2:
            categories = ['Financial', 'Driving', 'Health', 'Age', 'Claims']
            vals = [min((debt_ratio/0.5)*100,100), min((traffic_tickets/3)*100,100), min((bmi/35)*100,100), min((age/75)*100,100), min((past_claims/3)*100,100)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]], fill='toself', line_color=g_color))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Risk Breakdown (Radar)")
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- 7. الـ Bar Chart (تأثير العناصر) ---
        st.markdown("---")
        st.subheader("🔍 Impact Analysis (The Equation Factors)")
        
        impact_map = {
            "Low Credit Score": (850 - credit_score) / 5.5,
            "Claims History": past_claims * 10,
            "Debt Stress": debt_ratio * 100,
            "Driving Infractions": traffic_tickets * 15,
            "Medical Risk": (bmi-18) + (25 if smoking else 0)
        }
        df_impact = pd.DataFrame(impact_map.items(), columns=['Factor', 'Weight']).sort_values('Weight', ascending=True)
        fig_bar = px.bar(df_impact, x='Weight', y='Factor', orientation='h', color='Weight', color_continuous_scale='Reds', title="What pushed the score up?")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- 8. الـ Key Points (Metrics) ---
        st.markdown("---")
        st.subheader("📝 Key Decision Points")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Debt Ratio", f"{int(debt_ratio*100)}%", "High" if debt_ratio > 0.4 else "Healthy")
        k2.metric("Credit Points", f"{credit_score}", "- Risk" if credit_score > 700 else "+ Risk", delta_color="inverse")
        k3.metric("Tickets", f"{traffic_tickets}", "Review" if traffic_tickets > 2 else "Clean")
        k4.metric("AI Confidence", f"{confidence:.1f}%")

else:
    st.info("👈 Enter data and click **Analyze** to generate the full dashboard.")