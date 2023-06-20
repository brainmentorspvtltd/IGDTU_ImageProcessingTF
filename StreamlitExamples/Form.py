import streamlit as st

st.title("Registration Form...")

with st.form("reg_form"):
    name = st.text_input("Enter your name")
    email = st.text_input("Enter your email")
    password = st.text_input("Enter a password", type="password")
    gender = st.radio("Select your gender", ("Male", "Female"))
    country = st.selectbox("Country",["India","China","USA","UK","France","Italy"])
    age = st.slider("Enter your age", min_value=18, max_value=50)
    st.write(age)

    submit = st.form_submit_button("Register with Us...")