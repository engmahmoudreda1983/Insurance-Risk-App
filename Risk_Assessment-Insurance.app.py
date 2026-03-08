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

        # --- 6. عرض النتيجة للمستخدم ---
        st.subheader("📊 Final Assessment Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if result == 'Low':
                st.success(f"### 🟢 Risk Level: **{result} Risk**\nCustomer is safe to insure.")
            elif result == 'Medium':
                st.warning(f"### 🟡 Risk Level: **{result} Risk**\nProceed with standard checks.")
            else:
                st.error(f"### 🔴 Risk Level: **{result} Risk**\nRequires manual underwriter review.")
                
        with col2:
            st.info(f"### 🎯 AI Confidence: **{confidence:.2f}%**\nBased on PCA feature extraction.")
            
        st.markdown("---")
        st.write("📋 **Customer Input Summary:**")
        st.dataframe(input_data)
else:
    st.info("👈 Please enter the customer's details in the sidebar and click **Analyze Risk Profile** to see the magic happen.")