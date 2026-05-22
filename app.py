import streamlit as st
import joblib

# Modelleri yükle
model = joblib.load('sentiment_model.pkl')
vec = joblib.load('vectorizer.pkl')

st.title("🐦 Twitter Duygu Analizi Paneli")
st.write("Bu uygulama, girilen metinlerin duygusunu (Pozitif/Negatif) analiz eder.")

user_input = st.text_input("Analiz edilecek tweeti girin:")

if st.button("Analiz Et"):
    if user_input:
        data = vec.transform([user_input])
        prediction = model.predict(data)
        result = "Pozitif" if prediction[0] == 1 else "Negatif"
        st.success(f"Analiz Sonucu: {result}")
    else:
        st.warning("Lütfen bir metin girin!")