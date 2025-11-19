import streamlit as st
import requests

API_URL = "http://localhost:8000/rag-inventory-search"

st.set_page_config(page_title="Inventory RAG Assistant", layout="wide")

st.title("📦 Inventory RAG Search Assistant")
st.write("Ask anything about your stock, inventory movement, stockouts, replenishment, etc.")

# Input box
question = st.text_area("Enter your question:", height=100, placeholder="Example: Which SKUs had stockout risk last month?")

if st.button("Search"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching inventory database..."):
            try:
                response = requests.post(API_URL, json={"question": question})

                if response.status_code == 200:
                    data = response.json()

                    st.subheader("🧠 Answer")
                    st.success(data["answer"])

                    st.subheader("📄 Relevant Source Documents")
                    for idx, src in enumerate(data["sources"], start=1):
                        with st.expander(f"Source {idx} — Metadata: {src['metadata']}"):
                            st.write(src["content"])

                else:
                    st.error(f"API Error: {response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error("Failed to connect to backend. Make sure FastAPI is running.")
                st.write(str(e))
