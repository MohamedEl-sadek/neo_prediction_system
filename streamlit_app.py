import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
from sklearn.preprocessing import StandardScaler, PolynomialFeatures # type: ignore
import warnings
warnings.filterwarnings('ignore')
import streamlit as st # type: ignore
st.write(f"Streamlit Version: {st.__version__}")


# Set page config
st.set_page_config(
    page_title="Near-Earth Objects Hazard Prediction",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .hazardous {
        background-color: #ffebee;
        color: #c62828;
    }
    .safe {
        background-color: #e8f5e8;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        with open('models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('poly.pkl', 'rb') as f:
            poly = pickle.load(f)
        return models, scaler, poly
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None, None

# Load sample data
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv('sample_test_data.csv')
        return df
    except FileNotFoundError:
        st.warning("Sample test data not found. You can still use manual input.")
        return None

def preprocess_input(data, scaler, poly):
    """Preprocess input data for prediction"""
    # Create estimated_diameter_mean
    data['estimated_diameter_mean'] = (data['estimated_diameter_min'] + data['estimated_diameter_max']) / 2
    
    # Apply polynomial features
    numerical_cols = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance', 'estimated_diameter_mean']
    poly_features = poly.transform(data[numerical_cols])
    poly_feature_names = poly.get_feature_names_out(numerical_cols)
    data_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=data.index)
    
    # Scale features
    data_scaled = scaler.transform(data_poly)
    
    return data_scaled

def main():
    st.markdown('<h1 class="main-header">🌌 Near-Earth Objects Hazard Prediction</h1>', unsafe_allow_html=True)
    
    # Load models
    models, scaler, poly = load_models()
    if models is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Single Prediction", "Batch Prediction", "Model Performance", "Data Exploration"])
    
    if page == "Single Prediction":
        single_prediction_page(models, scaler, poly)
    elif page == "Batch Prediction":
        batch_prediction_page(models, scaler, poly)
    elif page == "Model Performance":
        model_performance_page(models)
    elif page == "Data Exploration":
        data_exploration_page()

def single_prediction_page(models, scaler, poly):
    st.header("🎯 Single Object Prediction")
    st.write("Enter the characteristics of a Near-Earth Object to predict if it's hazardous.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        absolute_magnitude = st.number_input("Absolute Magnitude", value=20.0, min_value=5.0, max_value=35.0, step=0.1)
        estimated_diameter_min = st.number_input("Estimated Diameter Min (km)", value=0.1, min_value=0.001, max_value=100.0, step=0.001, format="%.3f")
        estimated_diameter_max = st.number_input("Estimated Diameter Max (km)", value=0.3, min_value=0.001, max_value=100.0, step=0.001, format="%.3f")
    
    with col2:
        relative_velocity = st.number_input("Relative Velocity (km/h)", value=50000.0, min_value=200.0, max_value=300000.0, step=100.0)
        miss_distance = st.number_input("Miss Distance (km)", value=40000000.0, min_value=6000.0, max_value=80000000.0, step=1000.0, format="%.0f")
    
    # Model selection
    selected_model = st.selectbox("Select Model", list(models.keys()))
    
    if st.button("Predict Hazard Level", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'absolute_magnitude': [absolute_magnitude],
            'estimated_diameter_min': [estimated_diameter_min],
            'estimated_diameter_max': [estimated_diameter_max],
            'relative_velocity': [relative_velocity],
            'miss_distance': [miss_distance]
        })
        
        # Preprocess and predict
        processed_data = preprocess_input(input_data, scaler, poly)
        prediction = models[selected_model].predict(processed_data)[0]
        prediction_proba = models[selected_model].predict_proba(processed_data)[0]
        
        # Display results
        if prediction:
            st.markdown(f'<div class="prediction-result hazardous">⚠️ HAZARDOUS OBJECT</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-result safe">✅ SAFE OBJECT</div>', unsafe_allow_html=True)
        
        # Display probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of being Safe", f"{prediction_proba[0]:.2%}")
        with col2:
            st.metric("Probability of being Hazardous", f"{prediction_proba[1]:.2%}")

def batch_prediction_page(models, scaler, poly):
    st.header("📊 Batch Prediction")
    st.write("Upload a CSV file or use sample data to predict multiple objects at once.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    # Load sample data option
    sample_data = load_sample_data()
    if sample_data is not None:
        if st.button("Use Sample Test Data"):
            uploaded_file = "sample"
    
    if uploaded_file is not None:
        if uploaded_file == "sample":
            df = sample_data.copy()
        else:
            df = pd.read_csv(uploaded_file)
        
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Model selection
        selected_model = st.selectbox("Select Model for Batch Prediction", list(models.keys()))
        
        if st.button("Run Batch Prediction", type="primary"):
            # Preprocess and predict
            processed_data = preprocess_input(df, scaler, poly)
            predictions = models[selected_model].predict(processed_data)
            prediction_probas = models[selected_model].predict_proba(processed_data)
            
            # Add results to dataframe
            results_df = df.copy()
            results_df['Predicted_Hazardous'] = predictions
            results_df['Probability_Safe'] = prediction_probas[:, 0]
            results_df['Probability_Hazardous'] = prediction_probas[:, 1]
            
            st.write("Prediction Results:")
            st.dataframe(results_df)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Objects", len(results_df))
            with col2:
                st.metric("Predicted Hazardous", sum(predictions))
            with col3:
                st.metric("Predicted Safe", len(predictions) - sum(predictions))
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="neo_predictions.csv",
                mime="text/csv"
            )

def model_performance_page(models):
    st.header("📈 Model Performance Comparison")
    st.write("Compare the performance of different models.")
    
    # Mock performance data (in a real scenario, you'd load this from saved metrics)
    performance_data = {
        'Random Forest': {'Accuracy': 0.952, 'Precision': 0.95, 'Recall': 0.96, 'F1-Score': 0.95},
        'XGBoost': {'Accuracy': 0.865, 'Precision': 0.88, 'Recall': 0.87, 'F1-Score': 0.86},
        'Logistic Regression': {'Accuracy': 0.890, 'Precision': 0.89, 'Recall': 0.89, 'F1-Score': 0.89},
        'SVM': {'Accuracy': 0.875, 'Precision': 0.87, 'Recall': 0.88, 'F1-Score': 0.87}
    }
    
    # Create performance comparison chart
    metrics_df = pd.DataFrame(performance_data).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('Score')
    ax.set_xlabel('Models')
    ax.legend(title='Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display metrics table
    st.write("Performance Metrics:")
    st.dataframe(metrics_df)

def data_exploration_page():
    st.header("🔍 Data Exploration")
    st.write("Explore the Near-Earth Objects dataset.")
    
    try:
        df = pd.read_csv('nearest-earth-objects(1910-2024).csv')
        
        # Basic statistics
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Hazardous Objects", sum(df['is_hazardous']))
        with col3:
            st.metric("Safe Objects", len(df) - sum(df['is_hazardous']))
        
        # Data preview
        st.subheader("Data Sample")
        st.dataframe(df.head(10))
        
        # Distribution plots
        st.subheader("Feature Distributions")
        
        numerical_columns = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']
        selected_column = st.selectbox("Select column to visualize", numerical_columns)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(df[selected_column].dropna(), bins=50, alpha=0.7)
        ax1.set_title(f'Distribution of {selected_column}')
        ax1.set_xlabel(selected_column)
        ax1.set_ylabel('Frequency')
        
        # Box plot by hazard status
        df_clean = df.dropna(subset=[selected_column])
        hazardous_data = df_clean[df_clean['is_hazardous']][selected_column]
        safe_data = df_clean[~df_clean['is_hazardous']][selected_column]
        
        ax2.boxplot([safe_data, hazardous_data], labels=['Safe', 'Hazardous'])
        ax2.set_title(f'{selected_column} by Hazard Status')
        ax2.set_ylabel(selected_column)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation matrix
        st.subheader("Feature Correlations")
        corr_matrix = df[numerical_columns].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure the CSV file is in the correct location.")

if __name__ == "__main__":
    main()

