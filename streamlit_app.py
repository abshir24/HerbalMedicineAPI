import streamlit as st
import requests

# App title
st.set_page_config(page_title="Herbal Medicine Assistant", layout="centered")
st.title("🌿 Herbal Medicine Assistant")
st.markdown("Ask any question about herbs and their healing properties.")

# Input field
user_query = st.text_area("🔍 Enter your question:", placeholder="e.g. What herbs help with fibroids?")

# Submit button
if st.button("Get Herbal Guidance"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🧠 Consulting the herbal archives..."):
            try:
                response = requests.post(
                    "https://herbalmedicineapi.onrender.com/query",
                    json={"query": user_query},
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()

                # Show main answer
                st.subheader("🌱 Main Answer")
                st.write(data.get("main_answer", "No answer found."))

                # Show additional notes if available
                if data.get("additional_notes") and data["additional_notes"].strip().lower() != "no additional notes provided.":
                    st.subheader("📌 Additional Notes")
                    st.write(data["additional_notes"])

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
