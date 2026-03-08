import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Insurance Risk AI Assessor", page_icon="🛡️", layout="wide")

# --- 2. تحميل النماذج (باستخدام Cache لتسريع التطبيق) ---
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_model.pkl')
    le = joblib.load('label_encoder.pkl')
    xgb_model = joblib.load('xgboost_risk_model.pkl')
    return scaler, pca, le, xgb_model

scaler, pca, le, xgb_model = load_models()

# --- 3. تصميم رأس الصفحة ---
st.title("🛡️ AI-Driven Insurance Risk Assessment")
st.markdown("""
This enterprise application utilizes **Principal Component Analysis (PCA)** for dimensionality reduction 
and **eXtreme Gradient Boosting (XGBoost)** to evaluate customer risk profiles instantly.
""")
st.markdown("---")

# --- 4. الشريط الجانبي (Sidebar) لإدخال بيانات العميل ---
st.sidebar.header("📝 Customer Information")

# قسمنا الـ 20 متغير عشان الشاشة تكون منظمة
with st.sidebar.expander("👤 Personal Details", expanded=True):
    age = st.slider("Age", 18, 75, 35)
    dependents = st.number_input("Dependents", 0, 5, 1)
    edu_level = st.selectbox("Education Level (1=High School, 4=PhD)", [1, 2, 3, 4], index=1)

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
    blood_pressure = st.slider("Blood Pressure (Sys)", 90, 200, 120)

with st.sidebar.expander("📄 Insurance & Asset History", expanded=False):
    tenure = st.slider("Policy Tenure (Months)", 0, 240, 24)
    past_claims = st.number_input("Past Claims Count", 0, 10, 0)
    claims_amount = st.number_input("Total Claims Amount ($)", 0, 50000, 0)
    traffic_tickets = st.number_input("Traffic Tickets", 0, 10, 0)
    missed_payments = st.number_input("Missed Payments", 0, 10, 0)
    property_val = st.number_input("Property Value ($)", 50000, 1000000, 200000)
    vehicle_age = st.slider("Vehicle Age (Years)", 0, 30, 5)
    commute = st.slider("Daily Commute (Miles)", 0, 150, 20)

# --- 5. زر التوقع ومعالجة البيانات ---
if st.sidebar.button("🔍 Analyze Risk Profile", use_container_width=True):
    with st.spinner('Applying PCA and calculating risk with XGBoost...'):
        # تجميع الـ 20 متغير في DataFrame بنفس ترتيب التدريب
        input_data = pd.DataFrame({
            'Age': [age], 'Dependents': [dependents], 'Education_Level': [edu_level],
            'Annual_Income': [income], 'Credit_Score': [credit_score], 'Debt_to_Income_Ratio': [debt_ratio],
            'Savings_Amount': [savings], 'BMI': [bmi], 'Smoking_Status': [smoking],
            'Chronic_Diseases': [chronic], 'Exercise_Days_Per_Week': [exercise],
            'Blood_Pressure_Sys': [blood_pressure], 'Policy_Tenure_Months': [tenure],
            'Past_Claims_Count': [past_claims], 'Total_Claims_Amount': [claims_amount],
            'Traffic_Tickets': [traffic_tickets], 'Missed_Payments': [missed_payments],
            'Property_Value': [property_val], 'Vehicle_Age': [vehicle_age], 'Daily_Commute_Miles': [commute]
        })

        # خط الإنتاج (Pipeline): وزن -> ضغط -> توقع
        input_scaled = scaler.transform(input_data)  # التوزين
        input_pca = pca.transform(input_scaled)      # الضغط لـ 18 عامل
        prediction_encoded = xgb_model.predict(input_pca) # التوقع بالأرقام
        prediction_proba = xgb_model.predict_proba(input_pca) # نسبة الثقة
        
        # ترجمة النتيجة لنص
        result = le.inverse_transform(prediction_encoded)[0]
        confidence = np.max(prediction_proba) * 100

       # --- 6. عرض النتيجة والرسومات للمستخدم ---
        st.markdown("---")
        st.subheader("📊 Visual Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        # تحديد الألوان وقيمة العداد بناءً على النتيجة
        if result == 'Low':
            gauge_val = 15
            gauge_color = "#00CC96" # أخضر
            status_text = "Safe"
            st.success("### 🟢 Customer is safe to insure.")
        elif result == 'Medium':
            gauge_val = 50
            gauge_color = "#FECB52" # أصفر
            status_text = "Warning"
            st.warning("### 🟡 Proceed with standard checks.")
        else:
            gauge_val = 85
            gauge_color = "#EF553B" # أحمر
            status_text = "Risky"
            st.error("### 🔴 Requires manual underwriter review.")

        # 1. رسمة العداد (Gauge Chart)
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

        # 2. رسمة الشبكة العنكبوتية (Radar Chart)
        with col2:
            # تحجيم القيم من 0 لـ 100 تقريبياً عشان الرسمة تطلع متناسقة
            categories = ['Financial Stress', 'Driving Risk', 'Health (BMI)', 'Age Factor', 'Claims History']
            values = [
                min(debt_ratio * 100, 100),               # الديون
                min((traffic_tickets / 5) * 100, 100),    # المخالفات
                min((bmi / 40) * 100, 100),               # الوزن
                min((age / 75) * 100, 100),               # العمر
                min((past_claims / 5) * 100, 100)         # المطالبات
            ]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
              r=values + [values[0]], # قفل الدائرة
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
            
        st.info(f"🎯 **AI Confidence Level:** {confidence:.2f}% (Powered by PCA & XGBoost)")