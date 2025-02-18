import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple

class PreprocessingService:
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Complete preprocessing pipeline"""
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Store original data info
        preprocessing_info = {
            'original_shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'numeric_columns': list(df.select_dtypes(include=['int64', 'float64']).columns)
        }
        
        # Handle missing values
        numeric_cols = preprocessing_info['numeric_columns']
        categorical_cols = preprocessing_info['categorical_columns']
        
        if numeric_cols:
            df[numeric_cols] = self.numeric_imputer.fit_transform(df[numeric_cols])
        if categorical_cols:
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        
        # Encode categorical variables
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Scale numeric features
        if numeric_cols:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        preprocessing_info['final_shape'] = df.shape
        
        return df, preprocessing_info 