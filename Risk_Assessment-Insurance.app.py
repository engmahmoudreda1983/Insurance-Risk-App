import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Smart Risk Assessor", page_icon="🛡️", layout="wide")

# --- 2. تحميل النماذج (تأكد من وجود الملفات في نفس الفولدر) ---
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_model.pkl')
    le = joblib.load('label_encoder.pkl')
    xgb_model = joblib.load('xgboost_risk_model.pkl')
    return scaler, pca, le, xgb_model

try:
    scaler, pca, le, xgb_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please check your .pkl files.")

# --- 3. تصميم رأس الصفحة ---
st.title("🛡️ Smart Insurance Risk Assessment")
st.markdown("Comprehensive Underwriting Dashboard: Multi-Factor AI Risk Analysis")
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
    tenure = st.slider("Policy Tenure (Months)", 0, 240, 24)
    past_claims = st.number_input("Past Claims Count", 0, 10, 0)
    traffic_tickets = st.number_input("Traffic Tickets", 0, 10, 0)
    missed_payments = st.number_input("Missed Payments", 0, 10, 0)

# قيم افتراضية للحفاظ على استقرار الموديل
claims_amount, property_val, vehicle_age, commute_km, edu_level, bp_sys = 0, 200000, 5, 30, 2, 120

# --- 5. زر التوقع ومعالجة البيانات ---
if st.sidebar.button("🔍 Analyze Risk Profile", use_container_width=True):
    # تجهيز الداتا للموديل
    input_data = pd.DataFrame([[age, dependents, edu_level, income, credit_score, debt_ratio, savings, bmi, smoking, chronic, exercise, bp_sys, tenure, past_claims, claims_amount, traffic_tickets, missed_payments, property_val, vehicle_age, (commute_km * 0.621)]], 
                               columns=['Age', 'Dependents', 'Education_Level', 'Annual_Income', 'Credit_Score', 'Debt_to_Income_Ratio', 'Savings_Amount', 'BMI', 'Smoking_Status', 'Chronic_Diseases', 'Exercise_Days_Per_Week', 'Blood_Pressure_Sys', 'Policy_Tenure_Months', 'Past_Claims_Count', 'Total_Claims_Amount', 'Traffic_Tickets', 'Missed_Payments', 'Property_Value', 'Vehicle_Age', 'Daily_Commute_Miles'])

    # Processing
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    prediction_proba = xgb_model.predict_proba(input_pca)[0] 
    
    # تحديد النتيجة واللون
    classes = le.classes_
    proba_dict = dict(zip(classes, prediction_proba))
    result = classes[np.argmax(prediction_proba)]
    confidence = np.max(prediction_proba) * 100
    
    # تصحيح معادلة العداد (Low = 15%, Medium = 50%, High = 90%)
    gauge_val = int((proba_dict.get('Low', 0) * 15) + (proba_dict.get('Medium', 0) * 50) + (proba_dict.get('High', 0) * 90))
    g_color = "#00CC96" if result == 'Low' else "#FECB52" if result == 'Medium' else "#EF553B"

    # --- 6. عرض الداش بورد (العداد والعنكبوت) ---
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
        radar_categories = ['Financial Stress', 'Driving Risk', 'Health (BMI)', 'Age Factor', 'Claims History']
        radar_vals = [min((debt_ratio/0.5)*100,100), min((traffic_tickets/3)*100,100), min((bmi/35)*100,100), min((age/75)*100,100), min((past_claims/3)*100,100)]
        fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals + [radar_vals[0]], theta=radar_categories + [radar_categories[0]], fill='toself', line_color=g_color))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Risk Breakdown (Radar)")
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- 7. رسمة الـ Bar Chart (تأثير كل عنصر بالظبط) ---
    st.markdown("---")
    st.subheader("🔍 Impact Analysis (The Equation Factors)")
    
    impact_map = {
        "Credit History": (850 - credit_score) / 5.5,
        "Claims Impact": past_claims * 10,
        "Debt-to-Income": debt_ratio * 100,
        "Traffic Tickets": traffic_tickets * 15,
        "Health & Smoking": (bmi-18) + (25 if smoking else 0)
    }
    df_impact = pd.DataFrame(list(impact_map.items()), columns=['Factor', 'Weight']).sort_values('Weight', ascending=True)
    fig_bar = px.bar(df_impact, x='Weight', y='Factor', orientation='h', color='Weight', 
                     color_continuous_scale='Reds', title="What factors pushed the risk higher?")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- 8. الـ Key Points (Metrics) ---
    st.markdown("---")
    st.subheader("📝 Key Decision Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Financial Debt", f"{int(debt_ratio*100)}%", "Critical" if debt_ratio > 0.4 else "Stable")
    k2.metric("Credit Score", f"{credit_score}", "- Risk" if credit_score > 700 else "+ Risk", delta_color="inverse")
    k3.metric("Tickets Count", f"{traffic_tickets}", "Review" if traffic_tickets > 2 else "Safe")
    k4.metric("AI Confidence", f"{confidence:.2f}%")

else:
    st.info("👈 Enter customer details and click **Analyze Risk Profile** to start.")