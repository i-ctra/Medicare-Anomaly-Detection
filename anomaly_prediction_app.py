import streamlit as st
import pandas as pd
from joblib import load

# -----------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------
st.set_page_config(page_title="Medicare Anomaly Detection", layout="wide")

# -----------------------------------------------------------------
# Custom CSS Styling
# -----------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 0.5rem 1rem;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------
# Load Model and Assets
# -----------------------------------------------------------------
def load_assets():
    """
    Load your model (e.g., Random Forest) and any label encoders or scalers as needed.
    """
    try:
        model_ = load('anomalyclassifier.pkl')               # The trained anomaly detection model
        encoder_ = load('label_encoders.pkl')       # The saved label encoders, if applicable
        # scaler_ = load('scaler.pkl')              # Uncomment if you also have a scaler
        return model_, encoder_  # add scaler_ if you load it
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        st.stop()

# -----------------------------------------------------------------
# Global Feature Lists
# -----------------------------------------------------------------
DEMOGRAPHIC_FEATURES = ["age", "gender", "income"]
PROCEDURE_RANGE = 130  # if you have proc_1..proc_130
PROC_COLS = [f"proc_{i}" for i in range(1, PROCEDURE_RANGE + 1)]

# -----------------------------------------------------------------
# Helper: If a feature is label-encoded or numeric
# -----------------------------------------------------------------
def get_dropdown_or_number(feature_name: str, encoders_dict, default_val=0):
    """
    If a label encoder exists for feature_name, show a dropdown with its classes.
    Otherwise, prompt for a numeric input.
    """
    if feature_name in encoders_dict:
        options = list(encoders_dict[feature_name].classes_)
        selection = st.selectbox(f"Select {feature_name}", options)
        return encoders_dict[feature_name].transform([selection])[0]
    else:
        return st.number_input(f"Enter {feature_name}", value=default_val)

# -----------------------------------------------------------------
# Sidebar: Collect Inputs
# -----------------------------------------------------------------
def sidebar_input(encoders):
    """
    Collect user inputs for:
      - Demographic (age, gender, income)
      - Procedure columns (1..130)
      - Summation of procedure counts stored in 'count'
    Returns a dict with all inputs.
    """
    st.sidebar.header("Patient Information")

    # Demographic features
    user_input = {}
    with st.sidebar.expander("Demographic Features", expanded=True):
        for f in DEMOGRAPHIC_FEATURES:
            user_input[f] = get_dropdown_or_number(f, encoders)

    # Let user pick which procedure codes were performed
    st.sidebar.subheader("Procedure Codes")
    selected_proc = st.sidebar.multiselect(
        "Select which procedure codes were performed (1..130):",
        options=[str(i) for i in range(1, PROCEDURE_RANGE + 1)],
        help="For each selected code, specify how many times it was performed."
    )

    total_count = 0
    proc_counts = {}
    # For each selected code, ask the user for a numeric count
    for code_str in selected_proc:
        c = st.sidebar.number_input(
            f"Count for proc_{code_str}",
            value=0, min_value=0
        )
        proc_counts[code_str] = c
        total_count += c

    # Fill procedure features in user_input
    for i in range(1, PROCEDURE_RANGE + 1):
        col_name = f"proc_{i}"
        if str(i) in proc_counts:
            user_input[col_name] = proc_counts[str(i)]
        else:
            user_input[col_name] = 0

    # The 'count' feature is automatically the sum of all procedure counts
    user_input['count'] = total_count

    # Display total count in sidebar
    st.sidebar.write(f"**Total Count:** {total_count}")

    return user_input

# -----------------------------------------------------------------
# Main Prediction Logic
# -----------------------------------------------------------------
def preprocess_and_predict(model, input_dict):
    """
    Convert input_dict into a DataFrame, align columns with the model, 
    optionally apply scaling, then predict.
    """
    # Convert to single-row DataFrame
    df_input = pd.DataFrame(input_dict, index=[0])

    # Reindex to align with model's training columns
    try:
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
    except AttributeError:
        st.error("Model does not have 'feature_names_in_' attribute. "
                 "Ensure the model was trained with named columns.")
        st.stop()

    # If you have a scaler, apply it here:
    # df_input[scaler.feature_names_in_] = scaler.transform(df_input[scaler.feature_names_in_])

    # Predict
    pred_class = model.predict(df_input)[0]         # 0 or 1
    pred_prob = model.predict_proba(df_input)[0][1] # Probability for anomaly (class 1)
    return pred_class, pred_prob

# -----------------------------------------------------------------
# Main App
# -----------------------------------------------------------------
def main():
    # 1. Load your model and encoders
    model, encoder = load_assets()

    # 2. Collect user inputs (via sidebar)
    user_data = sidebar_input(encoder)

    # 3. Main Page UI
    st.title("🩺 Medicare Anomaly Detection")
    st.write("### Predict the likelihood of a patient being anomalous based on demographics and procedure data.")
    st.write("**Instructions:** Provide the required information in the **sidebar** on the left, then click **Predict 🚀** below.")

    if st.button("Predict 🚀"):
        # 4. Prediction
        pred, prob = preprocess_and_predict(model, user_data)

        # 5. Display result
        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"Prediction: Anomalous Patient ⚠️\nProbability: {prob * 100:.2f}%")
        else:
            st.success(f"Prediction: Normal Patient ✅\nProbability: {prob * 100:.2f}%")

# -----------------------------------------------------------------
if __name__ == "__main__":
    main()
