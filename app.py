import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="🏥 Healthcare Dashboard", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
    .stApp {
        background-color: black;
    }
            .sidebar .sidebar-content {
        color: black;
    }
    h1, h2, h3, h4 {
        color: #333333;
    }
    [data-testid="stSidebar"] {
        background-color: #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 Hospital Management System Dashboard")
st.write("Welcome to the Hospital Management System Dashboard! This dashboard provides insights into hospital operations, patient care, and financial performance.")
st.write("Date and Time: ", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))


st.sidebar.header("📂 Upload Your Data Files")
dy_appoinments = st.sidebar.file_uploader("Appointments CSV", type=["csv"])
dy_billing = st.sidebar.file_uploader("Billing CSV", type=["csv"])
dy_doctors = st.sidebar.file_uploader("Doctors CSV", type=["csv"])
dy_patients = st.sidebar.file_uploader("Patients CSV", type=["csv"])
dy_treatments = st.sidebar.file_uploader("Treatments CSV", type=["csv"])


appointments = pd.read_csv('data/appointments.csv') if dy_appoinments is None else pd.read_csv(dy_appoinments)
billing = pd.read_csv('data/billing.csv') if dy_billing is None else pd.read_csv(dy_billing)
doctors = pd.read_csv('data/doctors.csv') if dy_doctors is None else pd.read_csv(dy_doctors)
patients = pd.read_csv('data/patients.csv') if dy_patients is None else pd.read_csv(dy_patients)
treatments = pd.read_csv('data/treatments.csv') if dy_treatments is None else pd.read_csv(dy_treatments)


appointments['appointment_date'] = pd.to_datetime(appointments['appointment_date'])
billing['bill_date'] = pd.to_datetime(billing['bill_date'])
treatments['treatment_date'] = pd.to_datetime(treatments['treatment_date'])
billing['amount'] = billing['amount'].astype(float)
treatments['cost'] = treatments['cost'].astype(float)


col1, col2, col3 = st.columns(3)
col1.metric("📅 Appointments", appointments.shape[0])
col2.metric("🧑‍⚕️ Doctors", doctors.shape[0])
col3.metric("💸 Revenue (₹)", f"{billing['amount'].sum():,.2f}")

st.divider()


tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "📈 Revenue & Prediction", "📋 Raw Data","AI cost Estimator"])

with tab1:
    st.subheader("📌 Reasons for Visits")
    reason_counts = appointments['reason_for_visit'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.bar(reason_counts.index, reason_counts.values, color='skyblue')
    ax1.set_title('Reasons for Visits')
    ax1.set_xlabel('Reason for Visit')
    ax1.set_xticklabels(reason_counts.index, rotation=30)
    st.pyplot(fig1, use_container_width=True)

    st.subheader("👨‍⚕️ Doctors by Specialization")
    spec_counts = doctors['specialization'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(spec_counts.index, spec_counts.values, color='salmon')
    ax2.set_title('Doctors per Specialization')
    ax2.set_xticklabels(spec_counts.index, rotation=30)
    st.pyplot(fig2, use_container_width=True)

    

    st.subheader("💳 Payment Methods")
    pay_counts = billing['payment_method'].value_counts()
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.bar(pay_counts.index, pay_counts.values, color='orange')
    ax3.set_title('Payment Methods')
    ax3.set_xticklabels(pay_counts.index, rotation=30)
    st.pyplot(fig3, use_container_width=True)

    st.subheader("💉 Treatments Overview")
    treat_counts = treatments['treatment_type'].value_counts()
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    ax4.bar(treat_counts.index, treat_counts.values, color='green')
    ax4.set_title('Treatment Types')
    ax4.set_xticklabels(treat_counts.index, rotation=30)
    st.pyplot(fig4, use_container_width=True)

    

    st.subheader("🏆 Top 5 Doctors (by Appointments)")
    appt_doctors = pd.merge(appointments, doctors, on='doctor_id', how='left')
    top_doctors = appt_doctors['first_name'].value_counts().head(5)
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    ax5.bar(top_doctors.index, top_doctors.values, color='purple')
    ax5.set_title('Top Doctors')
    st.pyplot(fig5, use_container_width=True)

    

    st.subheader("📈 Appointment trends over months")
    appt_trends = appointments['appointment_date'].dt.to_period('M').value_counts().sort_index()
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    ax6.plot(appt_trends.index.astype(str), appt_trends.values, marker='o')
    ax6.set_title('Appointment Trends')
    ax6.set_xlabel('Month')
    ax6.set_ylabel('Number of Appointments')
    st.pyplot(fig6, use_container_width=True)

    st.subheader("🚫 Appointment Status Overview")
    no_show = appointments[appointments['status'] == 'No-show'].shape[0]
    cancelled = appointments[appointments['status'] == 'Cancelled'].shape[0]
    total = appointments.shape[0]
    no_show_pct = (no_show / total) * 100
    cancelled_pct = (cancelled / total) * 100
    c1, c2 = st.columns(2)
    c1.metric("❌ No-Show Rate", f"{no_show_pct:.2f}%")
    c2.metric("🛑 Cancelled Rate", f"{cancelled_pct:.2f}%")

    st.subheader("Top 5 doctors by Appointments")
    top_doctors = appointments['doctor_id'].value_counts().head(5)
    top_doctors = top_doctors.reset_index().rename(columns={'index': 'doctor_id', 'doctor_id': 'appointment_count'})

    st.download_button(
        label="Download Top Doctors Data",
        data=top_doctors.to_csv(index=False).encode('utf-8'),
        file_name='top_doctors.csv',    
        mime='text/csv'
    )

with tab2:
    st.subheader("📈 Monthly Revenue: Actual vs Predicted")

    billing['month'] = billing['bill_date'].dt.to_period('M')
    revenue = billing.groupby('month')['amount'].sum().reset_index()
    revenue['month_num'] = np.arange(len(revenue))
    revenue['month_str'] = revenue['month'].astype(str)

    X = revenue[['month_num']]
    y = revenue['amount']
    model = LinearRegression()
    model.fit(X, y)
    revenue['predicted_amount'] = model.predict(X)

    fig6, ax6 = plt.subplots(figsize=(8, 4))
    ax6.plot(revenue['month_str'], revenue['amount'], marker='o', label='Actual')
    ax6.plot(revenue['month_str'], revenue['predicted_amount'], linestyle='--', label='Predicted')
    ax6.set_title('Revenue Trend')
    ax6.set_xlabel('Month')
    ax6.set_ylabel('Revenue (₹)')
    ax6.legend()
    st.pyplot(fig6, use_container_width=True)

    st.subheader("Estimated Cost of Treatments")
    treatment_costs = treatments.groupby('treatment_type')['cost'].sum().reset_index()
    treatment_costs = treatment_costs.sort_values(by='cost', ascending=False)

    model = LinearRegression()
    X = np.arange(len(treatment_costs)).reshape(-1, 1)
    y = treatment_costs['cost'].values
    model.fit(X, y)
    predicted_costs = model.predict(X)
    fig7, ax7 = plt.subplots(figsize=(8, 4))
    ax7.bar(treatment_costs['treatment_type'], treatment_costs['cost'], color='teal', label='Actual Costs')
    ax7.plot(treatment_costs['treatment_type'], predicted_costs, color='red', linestyle='--', label='Predicted Costs')
    ax7.set_title('Estimated Cost of Treatments')
    ax7.set_xlabel('Treatment Type')
    ax7.set_ylabel('Cost (₹)')
    ax7.legend()
    st.pyplot(fig7, use_container_width=True)

    next_month_num = revenue['month_num'].max() + 1
    next_pred = model.predict([[next_month_num]])[0]
    st.metric("📊 Predicted Next Month Revenue", f"₹{next_pred:,.2f}")

with tab3:
    with st.expander("📄 Appointments Data"):
        st.dataframe(appointments.head())
    with st.expander("📄 Doctors Data"):
        st.dataframe(doctors.head())
    with st.expander("📄 Patients Data"):
        st.dataframe(patients.head())
    with st.expander("📄 Treatments Data"):
        st.dataframe(treatments.head())
    with st.expander("📄 Billing Data"):
        st.dataframe(billing.head())




data = pd.merge(treatments, appointments, on='appointment_id', how='left')
data = pd.merge(data, doctors, on='doctor_id', how='left')
data = pd.merge(data, patients, on='patient_id', how='left')
data = pd.merge(data, billing[['treatment_id', 'payment_method']], on='treatment_id', how='left')


model_data = data[['treatment_type', 'payment_method', 'specialization', 'cost']].dropna()
model_data_dummy = pd.get_dummies(model_data, drop_first=True)
X = model_data_dummy.drop('cost', axis=1)
y = model_data_dummy['cost']
model = LinearRegression()
model.fit(X, y)
predicted_costs = model.predict(X)

with tab4:
    st.subheader("💸 AI Cost Estimator")
    st.write("Predict treatment cost based on Treatment Type, Payment Method, and Doctor Specialization.")

    with st.form("predict_cost_form"):
        treatment_type = st.selectbox("Treatment Type", model_data['treatment_type'].unique())
        payment_method = st.selectbox("Payment Method", model_data['payment_method'].unique())
        specialization = st.selectbox("Doctor Specialization", model_data['specialization'].unique())

        submit_button = st.form_submit_button("Predict Cost")

    if submit_button:
       
        input_data = {
            'treatment_type': [treatment_type],
            'payment_method': [payment_method],
            'specialization': [specialization]
        }

        input_df = pd.DataFrame(input_data)
        input_df = pd.get_dummies(input_df, drop_first=True)

       
        missing_cols = set(X.columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[X.columns] 

        predicted_cost = model.predict(input_df)
        st.success(f"💰 Estimated Treatment Cost: ₹{predicted_cost[0]:,.2f}")
