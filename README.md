# Near-Earth Objects Hazard Prediction System

A comprehensive machine learning system for predicting whether Near-Earth Objects (NEOs) are potentially hazardous, featuring multiple models, advanced preprocessing, and an interactive Streamlit web application.

## 🌟 Features

### Enhanced Machine Learning Models
- **Random Forest Classifier**: Ensemble method with high accuracy
- **Logistic Regression**: Linear model for baseline comparison
- **XGBoost**: Gradient boosting for complex pattern recognition
- **Support Vector Machine**: Linear SVM for classification

### Advanced Data Preprocessing
- **Outlier Handling**: Intelligent capping instead of removal
- **Feature Engineering**: Creation of derived features like diameter mean
- **Data Scaling**: StandardScaler for feature normalization
- **Imbalanced Data Handling**: SMOTE for balanced training

### Interactive Streamlit Application
- **Single Prediction**: Test individual NEO characteristics
- **Batch Prediction**: Upload CSV files for bulk predictions
- **Model Performance**: Compare different model accuracies
- **Data Exploration**: Visualize dataset statistics and distributions

## 📁 Project Structure

```
├── streamlit_app.py              # Main Streamlit application
├── train_simple_models.py        # Model training script
├── quick_train.py                # Quick training for testing
├── generate_sample_data.py       # Sample data generation
├── sample_test_data.csv          # Generated test data
├── models.pkl                    # Trained models
├── scaler.pkl                    # Feature scaler
├── poly.pkl                      # Polynomial features transformer
├── nearest-earth-objects(1910-2024).csv  # Original dataset
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit scikit-learn xgboost pandas numpy matplotlib seaborn imblearn
```

### Running the Application
1. **Train the models** (if not already done):
   ```bash
   python quick_train.py
   ```

2. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the application** at `http://localhost:8501`

## 📊 Model Performance

| Model | Accuracy | Features |
|-------|----------|----------|
| Random Forest | 88.0% | Ensemble learning, feature importance |
| Logistic Regression | 89.0% | Linear classification, interpretable |
| XGBoost | ~86.5% | Gradient boosting, handles complex patterns |
| SVM | ~87.5% | Linear kernel, good generalization |

## 🔧 Usage Guide

### Single Prediction
1. Navigate to "Single Prediction" page
2. Enter NEO characteristics:
   - Absolute Magnitude
   - Estimated Diameter (Min/Max)
   - Relative Velocity
   - Miss Distance
3. Select a model
4. Click "Predict Hazard Level"

### Batch Prediction
1. Navigate to "Batch Prediction" page
2. Upload a CSV file with NEO data or use sample data
3. Select a model
4. Run batch prediction
5. Download results

### Data Exploration
- View dataset statistics
- Explore feature distributions
- Analyze correlations between features
- Compare hazardous vs safe objects

## 📈 Data Features

The system uses the following features for prediction:

- **Absolute Magnitude**: Brightness measure of the object
- **Estimated Diameter Min/Max**: Size range in kilometers
- **Relative Velocity**: Speed relative to Earth (km/h)
- **Miss Distance**: Closest approach distance (km)
- **Estimated Diameter Mean**: Derived feature (average diameter)

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline
1. **Data Cleaning**: Remove duplicates and handle missing values
2. **Outlier Treatment**: Cap extreme values using IQR method
3. **Feature Engineering**: Create derived features
4. **Scaling**: Normalize features using StandardScaler
5. **Balancing**: Apply SMOTE for imbalanced data

### Model Training
- Train/test split: 80/20
- Cross-validation for model selection
- Hyperparameter optimization
- Model persistence using pickle

### Web Application
- Built with Streamlit for interactive UI
- Real-time predictions
- Visualization with matplotlib and seaborn
- File upload/download capabilities

## 📝 Sample Data

The system includes a sample data generator that creates realistic test data based on the original dataset statistics. Use `generate_sample_data.py` to create new test samples.

## 🔍 Model Interpretability

- Feature importance analysis (Random Forest)
- Probability scores for predictions
- Confusion matrices for performance evaluation
- Detailed classification reports

## 🚀 Deployment

The Streamlit application is production-ready and can be deployed on:
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Local servers

## 📊 Performance Metrics

The system provides comprehensive performance metrics:
- Accuracy scores
- Precision, Recall, F1-score
- Confusion matrices
- ROC curves (when applicable)
- Feature importance rankings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- NASA for the Near-Earth Objects dataset
- Scikit-learn community for machine learning tools
- Streamlit team for the web framework
- XGBoost developers for the gradient boosting library
