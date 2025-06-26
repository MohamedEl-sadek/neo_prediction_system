import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(['neo_id', 'name', 'orbiting_body'], axis=1, inplace=True)

    # Handle duplicates
    df.drop_duplicates(inplace=True)

    # Fill Null Values with median for numerical columns
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Feature Engineering: Create estimated_diameter_mean
    df['estimated_diameter_mean'] = (df['estimated_diameter_min'] + df['estimated_diameter_max']) / 2

    # Outlier handling: Cap outliers instead of removing rows
    for col in ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler

def train_models(X, y):
    # Handle imbalanced data using SMOTE
    smote = SMOTE(random_state=40)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=40)

    models = {}

    # Random Forest Model
    rf_model = RandomForestClassifier(random_state=40, n_estimators=50)
    rf_model.fit(x_train, y_train)
    models['Random Forest'] = rf_model

    # XGBoost Model
    xgb_model = XGBClassifier(random_state=40, n_estimators=50)
    xgb_model.fit(x_train, y_train)
    models['XGBoost'] = xgb_model

    # Logistic Regression Model
    lr_model = LogisticRegression(random_state=40, solver='liblinear')
    lr_model.fit(x_train, y_train)
    models['Logistic Regression'] = lr_model

    # Support Vector Machine Model (using a linear kernel for faster training)
    svm_model = SVC(kernel='linear', random_state=40, probability=True)
    svm_model.fit(x_train, y_train)
    models['SVM'] = svm_model

    return models, x_test, y_test

if __name__ == '__main__':
    df = pd.read_csv('nearest-earth-objects(1910-2024).csv')
    df_processed, scaler = preprocess_data(df.copy())

    X = df_processed.drop('is_hazardous', axis=1)
    y = df_processed['is_hazardous']

    trained_models, x_test, y_test = train_models(X, y)

    for name, model in trained_models.items():
        y_pred = model.predict(x_test)
        print(f'\n{name} Classification Report:')
        print(classification_report(y_test, y_pred))
        print(f'{name} Accuracy: {accuracy_score(y_test, y_pred)}')

    # Save models and scaler for later use in Streamlit
    import pickle
    with open('models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create a dummy poly object for compatibility
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=1, include_bias=False)  # Identity transformation
    poly.fit(X)
    with open('poly.pkl', 'wb') as f:
        pickle.dump(poly, f)
    
    print("\nModels saved successfully!")

