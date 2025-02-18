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
        
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, List[int]]:
        """Detect outliers using IQR method"""
        outliers = {}
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                outliers[col] = list(df[outlier_mask].index)
        return outliers
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.numeric_imputer.fit_transform(df[numeric_cols])
        
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
            
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
            
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using StandardScaler"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Complete preprocessing pipeline"""
        # Store original data info
        preprocessing_info = {
            'original_shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'numeric_columns': list(df.select_dtypes(include=['int64', 'float64']).columns)
        }
        
        # Apply preprocessing steps
        df = self.handle_missing_values(df)
        preprocessing_info['outliers'] = self.detect_outliers(
            df, preprocessing_info['numeric_columns']
        )
        df = self.encode_categorical_variables(df)
        df = self.scale_features(df)
        
        preprocessing_info['final_shape'] = df.shape
        
        return df, preprocessing_info 