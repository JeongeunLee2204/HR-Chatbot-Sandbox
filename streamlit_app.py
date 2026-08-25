"""
비개발자(HR팀)가 브라우저에서 써볼 수 있는 데모 UI.
FastAPI 서버가 떠 있어야 동작함 (main.py 먼저 실행).

실행: streamlit run streamlit_app.py
"""

import requests
import streamlit as st

st.title("사내 HR 챗봇 (데모)")

API_URL = "http://localhost:8000/chat"

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("질문을 입력하세요 (예: 재택근무 몇 번까지 가능해?)")

if st.button("전송") and question:
    response = requests.post(API_URL, json={"question": question})
    answer = response.json()["answer"]
    st.session_state.history.append((question, answer))

for q, a in reversed(st.session_state.history):
    st.markdown(f"**Q. {q}**")
    st.markdown(f"A. {a}")
    st.divider()
