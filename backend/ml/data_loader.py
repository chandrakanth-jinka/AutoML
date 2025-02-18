import pandas as pd
from typing import Tuple, Dict, Any
import os

class DataLoader:
    def __init__(self, data_dir: str = "backend/data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load data from CSV or Excel file and return basic summary"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load the file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
        # Generate data summary
        summary = {
            'columns': df.columns.tolist(),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
            'missing_values': {str(k): int(v) for k, v in df.isnull().sum().items()},
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        return df, summary
    
    def save_data(self, df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to the data directory"""
        if not filename.endswith(('.csv', '.xlsx')):
            filename += '.csv'
            
        file_path = os.path.join(self.data_dir, filename)
        
        if filename.endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)
            
        return file_path 