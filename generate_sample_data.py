import pandas as pd
import numpy as np

def generate_sample_test_data(n_samples=100):
    """Generate sample test data based on the original dataset structure"""
    
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic data based on realistic ranges from the original dataset
    data = {
        'absolute_magnitude': np.random.normal(22.9, 2.9, n_samples),
        'estimated_diameter_min': np.random.lognormal(-2.5, 1.5, n_samples),
        'estimated_diameter_max': np.random.lognormal(-1.5, 1.5, n_samples),
        'relative_velocity': np.random.normal(51000, 26000, n_samples),
        'miss_distance': np.random.normal(41500000, 20700000, n_samples)
    }
    
    # Ensure realistic constraints
    data['absolute_magnitude'] = np.clip(data['absolute_magnitude'], 9.0, 34.0)
    data['estimated_diameter_min'] = np.clip(data['estimated_diameter_min'], 0.001, 40.0)
    data['estimated_diameter_max'] = np.clip(data['estimated_diameter_max'], 0.001, 85.0)
    data['relative_velocity'] = np.clip(data['relative_velocity'], 200, 292000)
    data['miss_distance'] = np.clip(data['miss_distance'], 6000, 75000000)
    
    # Ensure diameter_max >= diameter_min
    for i in range(n_samples):
        if data['estimated_diameter_max'][i] < data['estimated_diameter_min'][i]:
            data['estimated_diameter_max'][i] = data['estimated_diameter_min'][i] * np.random.uniform(1.1, 3.0)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic object names and IDs
    df['neo_id'] = [f"TEST{1000000 + i}" for i in range(n_samples)]
    df['name'] = [f"Test Object {i+1}" for i in range(n_samples)]
    df['orbiting_body'] = ['Earth'] * n_samples
    
    # Reorder columns to match original dataset
    df = df[['neo_id', 'name', 'absolute_magnitude', 'estimated_diameter_min', 
             'estimated_diameter_max', 'orbiting_body', 'relative_velocity', 'miss_distance']]
    
    return df

if __name__ == '__main__':
    # Generate sample data
    sample_df = generate_sample_test_data(100)
    
    # Save to CSV
    sample_df.to_csv('sample_test_data.csv', index=False)
    
    print("Sample test data generated successfully!")
    print(f"Generated {len(sample_df)} test samples")
    print("\nFirst 5 rows:")
    print(sample_df.head())
    
    print("\nData summary:")
    print(sample_df.describe())

