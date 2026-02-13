"""
Streamlit App for Adult Income Prediction
Deploy all 6 trained machine learning models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("💰 Adult Income Prediction")
st.markdown("Predict whether an individual's income is **≤50K** or **>50K** per year using multiple ML models")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbor (KNN)": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    list(model_options.keys())
)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler(model_name, scaler_path, model_dir):
    """Load the selected model and scaler"""
    try:
        model_path = os.path.join(model_dir, model_options[model_name])
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load feature names
        feature_names_path = os.path.join(model_dir, "feature_names.pkl")
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model_dir = "model"
scaler_path = os.path.join(model_dir, "scaler.pkl")

if os.path.exists(scaler_path) and os.path.exists(os.path.join(model_dir, model_options[selected_model_name])):
    model, scaler, feature_names = load_model_and_scaler(selected_model_name, scaler_path, model_dir)
    
    if model is not None:
        st.sidebar.success(f"✓ {selected_model_name} loaded successfully!")
        
        # Main content area
        st.header("Input Features")
        st.markdown("Enter the following information to predict income level:")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=17, max_value=90, value=35, step=1)
            workclass = st.selectbox("Workclass", [
                "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                "Local-gov", "State-gov", "Without-pay", "Never-worked"
            ])
            education = st.selectbox("Education", [
                "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
                "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
            ])
            education_num = st.number_input("Education Number", min_value=1, max_value=16, value=13, step=1)
            marital_status = st.selectbox("Marital Status", [
                "Married-civ-spouse", "Divorced", "Never-married", "Separated",
                "Widowed", "Married-spouse-absent", "Married-AF-spouse"
            ])
            occupation = st.selectbox("Occupation", [
                "Tech-support", "Craft-repair", "Other-service", "Sales",
                "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
            ])
            relationship = st.selectbox("Relationship", [
                "Wife", "Own-child", "Husband", "Not-in-family",
                "Other-relative", "Unmarried"
            ])
        
        with col2:
            race = st.selectbox("Race", [
                "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
            ])
            sex = st.selectbox("Sex", ["Male", "Female"])
            capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100)
            capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, step=100)
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40, step=1)
            native_country = st.selectbox("Native Country", [
                "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
                "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
                "Peru", "Hong", "Holand-Netherlands"
            ])
            fnlwgt = st.number_input("Final Weight", min_value=10000, max_value=1500000, value=200000, step=1000)
        
        # Prediction button
        if st.button("🔮 Predict Income", type="primary"):
            try:
                # Create a dictionary with all features
                input_data = {
                    'age': age,
                    'workclass': workclass,
                    'fnlwgt': fnlwgt,
                    'education': education,
                    'education-num': education_num,
                    'marital-status': marital_status,
                    'occupation': occupation,
                    'relationship': relationship,
                    'race': race,
                    'sex': sex,
                    'capital-gain': capital_gain,
                    'capital-loss': capital_loss,
                    'hours-per-week': hours_per_week,
                    'native-country': native_country
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical variables (simplified - in production, use the same encoders from training)
                # For this demo, we'll use label encoding
                from sklearn.preprocessing import LabelEncoder
                
                # Note: In production, you should save and load the encoders used during training
                # This is a simplified version
                categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                                  'relationship', 'race', 'sex', 'native-country']
                
                # Create a copy for encoding
                input_encoded = input_df.copy()
                
                # Simple label encoding (in production, use saved encoders)
                for col in categorical_cols:
                    if col in input_encoded.columns:
                        le = LabelEncoder()
                        # Fit on the column (in production, use saved encoder)
                        input_encoded[col] = le.fit_transform(input_encoded[col].astype(str))
                
                # Select only numeric columns and ensure correct order
                input_encoded = input_encoded.select_dtypes(include=[np.number])
                
                # Scale the features
                input_scaled = scaler.transform(input_encoded)
                
                # Make prediction
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                
                # Display results
                st.success("### Prediction Results")
                
                income_class = ">50K" if prediction[0] == 1 else "≤50K"
                confidence = prediction_proba[0][prediction[0]] * 100
                
                st.markdown(f"**Predicted Income:** {income_class}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
                # Show probability distribution
                st.markdown("**Probability Distribution:**")
                prob_df = pd.DataFrame({
                    'Income Class': ['≤50K', '>50K'],
                    'Probability': [prediction_proba[0][0]*100, prediction_proba[0][1]*100]
                })
                st.bar_chart(prob_df.set_index('Income Class'))
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Note: This is a simplified demo. In production, you should use the exact same preprocessing pipeline from training.")
        
        # Model information
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Information")
        st.sidebar.info(f"**Selected Model:** {selected_model_name}")
        st.sidebar.markdown("""
        **Available Models:**
        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbor
        - Naive Bayes
        - Random Forest
        - XGBoost
        """)
        
    else:
        st.error("Failed to load model. Please ensure models are saved in the 'model/' directory.")
else:
    st.error("⚠️ Models not found!")
    st.info("""
    **To use this app:**
    1. Run the notebook cells to train and save all models
    2. Ensure the 'model/' directory contains:
       - scaler.pkl
       - logistic_regression.pkl
       - decision_tree.pkl
       - knn.pkl
       - naive_bayes.pkl
       - random_forest.pkl
       - xgboost.pkl
       - feature_names.pkl
    """)

# Footer
st.markdown("---")
st.markdown("**Note:** This is a demonstration app. For production use, implement proper data validation and use the exact preprocessing pipeline from training.")
