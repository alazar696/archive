import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set up the look and feel of the web page
st.set_page_config(page_title="Cardio Risk Predictor", page_icon="🫀", layout="centered")

@st.cache_resource
def load_assets():
    """
    Loads the trained models and scaler from your computer.
    @st.cache_resource makes sure this only happens once to keep the app lightning fast!
    """
    assets = {}
    try:
        # Load the scaler
        if os.path.exists("models/scaler.pkl"):
            assets['scaler'] = joblib.load("models/scaler.pkl") 
        if os.path.exists("models/tuned_random_forest.pkl"):
            assets['rf'] = joblib.load("models/tuned_random_forest.pkl")
            
        # Load the SVM 
        if os.path.exists("models/svm_model_standard.pkl"):
            assets['svm'] = joblib.load("models/svm_model_weighted.pkl")
        if os.path.exists("models/svm_model_standard.pkl"):
            assets['svm_standard'] = joblib.load("models/svm_model_standard.pkl")
        if os.path.exists("models/kmeans_model.pkl"):
            assets['kmeans'] = joblib.load("models/kmeans_model.pkl")
        if os.path.exists("models/deep_learning_model.pkl"):
            assets['dl'] = joblib.load("models/deep_learning_model.pkl")
                    
        # Optional: Load Deep Learning (Skipped here to keep the app lightweight, 
        # but you can add it back if your professor wants to see it!)
            
        return assets
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def main():
    # --- UI HEADER ---
    st.title("🫀 Cardiovascular Disease Risk Predictor")
    st.markdown("""
    Welcome to the diagnostic dashboard. Enter the patient's physiological metrics below 
    to assess their risk of cardiovascular disease using our optimized Machine Learning models.
    """)

    # Load our AI models into memory
    assets = load_assets()
    if not assets or 'scaler' not in assets:
        st.warning("⚠️ Could not load models. Please ensure your 'models/' folder exists and contains 'scaler.pkl' and 'tuned_random_forest.pkl'.")
        return

    # --- PATIENT INPUT FORM ---
    st.header("📋 Patient Information")
    
    # We use 3 columns to make the UI look professional and organized
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", value=120)
        chol = st.number_input("Cholesterol", value=200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
        
    with col3:
        oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0, step=0.1)
        slope = st.selectbox("ST Segment Slope (slope)", options=[0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

    # --- PREDICTION ENGINE ---
    st.markdown("---")
    st.header("🤖 Machine Learning Diagnosis")
    
    # Let the user pick which model to use!
    model_choice = st.selectbox("Choose the Predictive Model:", ["Fine-Tuned Random Forest", "Support Vector Machine (SVM)", "Deep Learning (Optional)", "K-Means Clustering (Exploratory)"])

    # The big red predict button
    if st.button("Run AI Diagnostics", type="primary"):
        
        # Package the user's input exactly how the model expects it (matching the UCI Heart dataset)
        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        # Scale the data using the scaler we saved during Phase 2
        try:
            scaled_data = assets['scaler'].transform(input_data)
        except Exception as e:
            st.error(f"Error processing inputs. Feature mismatch: {e}")
            return

        # Run the prediction through the chosen model
        prediction = None
        if model_choice == "Random Forest" and 'rf' in assets:
            prediction = assets['rf'].predict(scaled_data)[0]
        elif model_choice == "Support Vector Machine (SVM) weighted" and 'svm' in assets:
            prediction = assets['svm'].predict(scaled_data)[0]
        elif model_choice == "Support Vector Machine (SVM)" and 'svm_standard' in assets:
            prediction = assets['svm_standard'].predict(scaled_data)[0]
        elif model_choice == "Deep Learning (Optional)" and 'dl' in assets:
            prediction = assets['dl'].predict(scaled_data)[0]
        elif model_choice == "K-Means Clustering (Exploratory)" and 'kmeans' in assets:
            prediction = assets['kmeans'].predict(scaled_data)[0]
        else:
            st.error("Selected model file is missing from the 'models/' folder.")
            return

        # Display the final result beautifully
        st.markdown("### 📊 Diagnostic Result:")
        if prediction == 1:
            st.error("🚨 **Elevated Risk Detected**")
            st.write("The model identified patterns associated with cardiovascular disease. Please recommend consulting a healthcare professional.")
        else:
            st.success("✅ **Low Risk Detected**")
            st.write("The model did not find significant patterns associated with cardiovascular disease. Keep up the healthy lifestyle!")

if __name__ == "__main__":
    main()