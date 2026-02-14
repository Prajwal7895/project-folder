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

# Load model, scaler, and label encoders
@st.cache_resource
def load_model_and_scaler(model_name, scaler_path, model_dir):
    """Load the selected model, scaler, feature names, and label encoders"""
    try:
        model_path = os.path.join(model_dir, model_options[model_name])
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load feature names
        feature_names_path = os.path.join(model_dir, "feature_names.pkl")
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load label encoders (CRITICAL for correct predictions!)
        label_encoders_path = os.path.join(model_dir, "label_encoders.pkl")
        le_dict = {}
        le_target = None
        if os.path.exists(label_encoders_path):
            with open(label_encoders_path, 'rb') as f:
                label_encoders = pickle.load(f)
            le_dict = label_encoders.get('feature_encoders', {})
            le_target = label_encoders.get('target_encoder', None)
        else:
            st.warning("⚠️ label_encoders.pkl not found! Predictions may be incorrect.")
        
        return model, scaler, feature_names, le_dict, le_target
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, {}, None

model_dir = "model"
scaler_path = os.path.join(model_dir, "scaler.pkl")

if os.path.exists(scaler_path) and os.path.exists(os.path.join(model_dir, model_options[selected_model_name])):
    model, scaler, feature_names, le_dict, le_target = load_model_and_scaler(selected_model_name, scaler_path, model_dir)
    
    if model is not None:
        st.sidebar.success(f"✓ {selected_model_name} loaded successfully!")
        if le_dict:
            st.sidebar.info(f"✓ Label encoders loaded ({len(le_dict)} feature encoders)")
        
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
                
                # Encode categorical variables using SAVED encoders from training
                categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                                  'relationship', 'race', 'sex', 'native-country']
                
                input_encoded = input_df.copy()
                
                # Use saved label encoders (CRITICAL for correct predictions!)
                if le_dict:
                    for col in categorical_cols:
                        if col in input_encoded.columns and col in le_dict:
                            le = le_dict[col]
                            try:
                                # Transform using saved encoder
                                input_encoded[col] = le.transform(input_encoded[col].astype(str))
                            except ValueError:
                                # If value not seen during training, use most common class
                                input_encoded[col] = le.transform([le.classes_[0]])[0]
                else:
                    # Fallback: create new encoders (WARNING: may cause incorrect predictions!)
                    st.warning("⚠️ Using new encoders instead of saved ones. Predictions may be incorrect!")
                    from sklearn.preprocessing import LabelEncoder
                for col in categorical_cols:
                    if col in input_encoded.columns:
                        le = LabelEncoder()
                        input_encoded[col] = le.fit_transform(input_encoded[col].astype(str))
                
                # Select only numeric columns and ensure correct order
                input_encoded = input_encoded.select_dtypes(include=[np.number])
                
                # Ensure columns are in the same order as training
                if feature_names:
                    # Reorder columns to match training data
                    missing_cols = set(feature_names) - set(input_encoded.columns)
                    if missing_cols:
                        st.error(f"Missing columns: {missing_cols}")
                    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
                
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
                st.exception(e)
        
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
       - label_encoders.pkl (CRITICAL!)
    """)

# Footer
st.markdown("---")
st.markdown("**Note:** This app uses the exact same preprocessing pipeline from training for accurate predictions.")
