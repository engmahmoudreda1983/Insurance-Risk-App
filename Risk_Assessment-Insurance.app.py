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
st.markdown("Instantly evaluate customer profiles using advanced AI with clear risk factor transparency.")
st.markdown("---")

# --- 4. الشريط الجانبي (Sidebar) ---
st.sidebar.header("📝 Customer Information")

with st.sidebar.expander("👤 Personal & Financial", expanded=True):
    age = st.slider("Age", 18, 75, 35)
    income = st.number_input("Annual Income ($)", 30000, 300000, 70000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)

with st.sidebar.expander("📄 History & Lifestyle", expanded=False):
    past_claims = st.number_input("Past Claims Count", 0, 10, 0)
    traffic_tickets = st.number_input("Traffic Tickets", 0, 10, 0)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")

# قيم افتراضية للحفاظ على توافق الموديل
dependents, edu_level, savings, chronic, exercise, bp_sys, tenure, claims_amount, missed_payments, property_val, vehicle_age, commute_km = 1, 2, 20000, 0, 3, 120, 24, 0, 0, 200000, 5, 30

# --- 5. زر التوقع ومعالجة البيانات ---
if st.sidebar.button("🔍 Analyze Risk Profile", use_container_width=True):
    with st.spinner('Generating comprehensive report...'):
        
        # تجهيز البيانات
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

        # --- 6. عرض النتيجة والرسومات القديمة ---
        st.subheader("📊 Visual Risk Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = gauge_val,
                title = {'text': f"Overall Risk: {result}", 'font': {'size': 24}},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': g_color}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # رسمة العنكبوت القديمة
            categories = ['Financial Stress', 'Driving Risk', 'Health (BMI)', 'Age Factor', 'Claims History']
            values = [min((debt_ratio/0.5)*100, 100), min((traffic_tickets/3)*100, 100), max(0, min(((bmi-18.5)/21.5)*100, 100)), max(0, min(((age-18)/57)*100, 100)), min((past_claims/3)*100, 100)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', line_color=g_color))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Customer Risk Breakdown")
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- 7. الجزء الجديد: Bar Chart (أساس المعادلة) ---
        st.markdown("---")
        st.subheader("🔍 Factor Impact Analysis (The Equation)")
        
        # حساب أوزان العناصر بناءً على المدخلات (تفسير الموديل)
        impact_map = {
            "Credit History": (850 - credit_score) / 5.5,
            "Claims Impact": past_claims * 10,
            "Debt Stress": debt_ratio * 100,
            "Driving Record": traffic_tickets * 15,
            "Lifestyle (BMI/Smoke)": (bmi-18.5) + (25 if smoking else 0)
        }
        df_impact = pd.DataFrame(impact_map.items(), columns=['Factor', 'Impact Weight']).sort_values('Impact Weight', ascending=True)
        
        fig_bar = px.bar(df_impact, x='Impact Weight', y='Factor', orientation='h', 
                         title="How each factor pushed the score up:",
                         color='Impact Weight', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.info(f"🎯 **AI Confidence Level:** {confidence:.2f}%")
else:
    st.info("👈 Enter details in the sidebar and click **Analyze** to generate the full report.")