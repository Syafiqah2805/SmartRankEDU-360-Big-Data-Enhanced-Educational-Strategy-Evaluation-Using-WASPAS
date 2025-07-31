# SmartRankEDU 360: Big Data-Enhanced Educational Strategy Evaluation Using WASPAS
# Streamlit app to perform MCDM using WASPAS

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SmartRankEDU 360", layout="wide")
st.title("📊 SmartRankEDU 360")
st.subheader("Big Data-Enhanced Educational Strategy Evaluation Using WASPAS")

st.markdown("""
This system helps evaluate and rank educational strategies or tools using the **WASPAS** method,
leveraging multi-criteria decision-making (MCDM) powered by educational data.
""")

# --- Upload Data ---
st.header("1. Upload Decision Matrix")
uploaded_file = st.file_uploader("Upload a CSV file with Alternatives × Criteria", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df, use_container_width=True)

    st.header("2. Set Criteria Type and Weights")
    with st.form("criteria_form"):
        criteria_names = df.columns[1:]  # Exclude alternative names
        criteria_types = []
        criteria_weights = []

        st.markdown("**Specify each criterion as 'Benefit' or 'Cost' and assign a weight (0-1).**")
        for c in criteria_names:
            col1, col2 = st.columns([2, 1])
            with col1:
                crit_type = st.selectbox(f"Type for {c}", ["Benefit", "Cost"], key=f"type_{c}")
            with col2:
                weight = st.number_input(f"Weight for {c}", min_value=0.0, max_value=1.0, value=1.0/len(criteria_names), key=f"weight_{c}")
            criteria_types.append(crit_type)
            criteria_weights.append(weight)

        submitted = st.form_submit_button("Apply")

    if submitted:
        st.success("Criteria types and weights saved. Calculating WASPAS rankings...")

        # --- Normalize the matrix ---
        decision_matrix = df.iloc[:, 1:].values
        norm_matrix = np.zeros_like(decision_matrix, dtype=float)

        for j, t in enumerate(criteria_types):
            if t == "Benefit":
                norm_matrix[:, j] = decision_matrix[:, j] / decision_matrix[:, j].max()
            else:
                norm_matrix[:, j] = decision_matrix[:, j].min() / decision_matrix[:, j]

        weights = np.array(criteria_weights)

        # --- WASPAS Calculation ---
        WSM = np.dot(norm_matrix, weights)
        WPM = np.prod(np.power(norm_matrix, weights), axis=1)
        WASPAS = 0.5 * WSM + 0.5 * WPM

        results = pd.DataFrame({
            "Alternative": df.iloc[:, 0],
            "WSM Score": WSM,
            "WPM Score": WPM,
            "WASPAS Score": WASPAS,
            "Rank": WASPAS.argsort().argsort() + 1
        }).sort_values("WASPAS Score", ascending=False).reset_index(drop=True)

        st.header("3. 🥇 Ranking Results")
        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "waspas_results.csv", "text/csv")

        st.header("4. 📈 Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="WASPAS Score", y="Alternative", data=results, palette="viridis", ax=ax)
        ax.set_title("Ranking of Alternatives by WASPAS Score")
        st.pyplot(fig)

        st.header("5. 📊 Score Comparison Table")
        st.dataframe(results[["Alternative", "WSM Score", "WPM Score", "WASPAS Score"]])

else:
    st.info("Please upload a CSV file to proceed. Example format: [Alternative, Criteria1, Criteria2, ...]")

st.markdown("---")
st.caption("© 2025 SmartRankEDU 360 — Built for X-RIC 2025")
