from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_auc_score, explained_variance_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

class ModelTrainer:
    def __init__(self):
        # Define default model parameters
        self.classification_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,  # Increased from default 100
                C=1.0,
                solver='lbfgs',
                multi_class='auto',
                random_state=42
            ),
            'svm': SVC(
                probability=True,
                kernel='rbf',
                C=1.0,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        self.regression_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'svr': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        self.clustering_models = {
            'kmeans': KMeans(
                n_clusters=3,  # default value, will be optimized
                random_state=42
            ),
            'dbscan': DBSCAN(
                eps=0.5,
                min_samples=5
            ),
            'hierarchical': AgglomerativeClustering(
                n_clusters=3
            )
        }
        
        self.trained_model = None
        self.model_type = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
        # Create plots directory if it doesn't exist
        self.plots_dir = Path("backend/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def _plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.plots_dir / 'confusion_matrix.png')
        plt.close()
        
    def _plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve"""
        plt.figure(figsize=(10, 8))
        
        if y_pred_proba.shape[1] > 2:  # Multi-class
            for i in range(y_pred_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        else:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.plots_dir / 'roc_curve.png')
        plt.close()
        
    def _plot_precision_recall_curve(self, y_true, y_pred_proba):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(10, 8))
        
        if y_pred_proba.shape[1] > 2:  # Multi-class
            for i in range(y_pred_proba.shape[1]):
                precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
                avg_precision = average_precision_score(y_true == i, y_pred_proba[:, i])
                plt.plot(recall, precision, label=f'Class {i} (AP = {avg_precision:.2f})')
        else:  # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
            plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(self.plots_dir / 'precision_recall_curve.png')
        plt.close()

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate relevant metrics based on problem type"""
        try:
            if self.model_type == 'classification':
                # Basic classification metrics
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                    'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                    'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                    'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                }
                
                # Per-class metrics
                classes = np.unique(y_true)
                for cls in classes:
                    cls_name = str(self.label_encoder.inverse_transform([cls])[0])
                    metrics[f'precision_class_{cls_name}'] = precision_score(y_true == cls, y_pred == cls, zero_division=0)
                    metrics[f'recall_class_{cls_name}'] = recall_score(y_true == cls, y_pred == cls, zero_division=0)
                    metrics[f'f1_class_{cls_name}'] = f1_score(y_true == cls, y_pred == cls, zero_division=0)
                
                # Generate plots if probabilities are available
                if y_pred_proba is not None:
                    try:
                        # Confusion Matrix
                        self._plot_confusion_matrix(y_true, y_pred, 
                                                  labels=self.label_encoder.classes_)
                        
                        # ROC Curve
                        self._plot_roc_curve(y_true, y_pred_proba)
                        if y_pred_proba.shape[1] == 2:  # Binary classification
                            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                            metrics['roc_auc'] = auc(fpr, tpr)
                        else:  # Multi-class
                            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, 
                                                                   multi_class='ovr', 
                                                                   average='macro')
                        
                        # Precision-Recall Curve
                        self._plot_precision_recall_curve(y_true, y_pred_proba)
                        
                    except Exception as e:
                        print(f"Warning: Could not generate some plots: {str(e)}")
                    
                return metrics
                
            else:  # regression
                return {
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred),
                    'explained_variance': explained_variance_score(y_true, y_pred)
                }
        except Exception as e:
            print(f"Warning: Error calculating some metrics: {str(e)}")
            return {'error': str(e)}

    def preprocess_for_training(self, X: pd.DataFrame, y: pd.Series, problem_type: str):
        """Preprocess data specifically for model training"""
        try:
            # Identify column types
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            # Create preprocessing pipelines
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='passthrough'
            )
            
            # Fit and transform the features
            X_processed = preprocessor.fit_transform(X)
            
            # Get feature names
            self.feature_names = []
            
            # Add numeric feature names
            if len(numeric_features) > 0:
                self.feature_names.extend(numeric_features)
            
            # Add categorical feature names
            if len(categorical_features) > 0:
                for feature in categorical_features:
                    encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                    encoded_features = encoder.get_feature_names_out([feature])
                    self.feature_names.extend(encoded_features)
            
            # Process target variable
            if problem_type == 'classification':
                # For classification, ensure consecutive integers starting from 0
                unique_labels = np.unique(y)
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                y_processed = np.array([label_map[label] for label in y])
                
                # Store label mapping for later use
                self.label_encoder.classes_ = unique_labels
            else:
                y_processed = y.astype(float)
            
            return X_processed, y_processed
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def _plot_distribution(self, df: pd.DataFrame, column: str, plots_dir: Path):
        """Safely plot distribution with proper file naming"""
        try:
            # Create a safe filename
            safe_filename = "".join(c if c.isalnum() else "_" for c in column)
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=column, kde=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(plots_dir / f'{safe_filename}_distribution.png')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create distribution plot for {column}: {str(e)}")

    def analyze_and_visualize_data(self, df: pd.DataFrame, summary: dict):
        """Analyze data and create visualizations with safe file handling"""
        plots_dir = Path("backend/plots/data_analysis")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n=== Dataset Analysis ===")
        print(f"Total rows: {summary['num_rows']}")
        print(f"Total columns: {summary['num_columns']}")
        
        # Data types and missing values
        print("\nColumn Information:")
        for col in df.columns:
            missing = summary['missing_values'].get(col, 0)
            dtype = summary['dtypes'].get(col, '')
            print(f"{col}:")
            print(f"  - Type: {dtype}")
            print(f"  - Missing values: {missing}")
            print(f"  - Unique values: {df[col].nunique()}")
            
            # Create distributions for numeric columns
            if col in summary['numeric_columns']:
                self._plot_distribution(df, col, plots_dir)
                
                # Basic statistics
                print(f"  - Mean: {df[col].mean():.2f}")
                print(f"  - Std: {df[col].std():.2f}")
                print(f"  - Min: {df[col].min():.2f}")
                print(f"  - Max: {df[col].max():.2f}")
            
            # Value counts for categorical columns
            if col in summary['categorical_columns']:
                print("  - Top categories:")
                for val, count in df[col].value_counts().head().items():
                    print(f"    {val}: {count}")
        
        # Correlation matrix for numeric columns
        if len(summary['numeric_columns']) > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df[summary['numeric_columns']].corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(plots_dir / 'correlation_matrix.png')
            plt.close()
        
        print(f"\nData visualizations saved in: {plots_dir}")
        return plots_dir

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        model_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        **model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a model with proper preprocessing"""
        try:
            # Validate inputs
            if problem_type not in ['classification', 'regression']:
                raise ValueError("problem_type must be 'classification' or 'regression'")
            
            # Select model
            models = (
                self.classification_models if problem_type == 'classification'
                else self.regression_models
            )
            
            if model_name not in models:
                raise ValueError(f"Model {model_name} not found for {problem_type}")
            
            # Preprocess data
            print("Preprocessing data...")
            X_processed, y_processed = self.preprocess_for_training(X, y, problem_type)
            
            # For classification, check if we need to handle many classes
            if problem_type == 'classification' and len(np.unique(y_processed)) > 1000:
                print("Warning: Large number of classes detected. Consider using regression instead.")
                if model_name == 'xgboost':
                    print("Switching to XGBRegressor due to large number of classes...")
                    model = xgb.XGBRegressor()
                    problem_type = 'regression'
            else:
                model = models[model_name]
            
            # Split data
            print("Splitting data into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=test_size, random_state=random_state
            )
            
            # Initialize and train model
            print(f"Training {model_name}...")
            if model_params:
                model.set_params(**model_params)
            
            model.fit(X_train, y_train)
            
            # Store trained model and type
            self.trained_model = model
            self.model_type = problem_type
            
            # Generate predictions
            print("Generating predictions...")
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if problem_type == 'classification' and hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            print("Calculating metrics...")
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Calculate cross-validation scores
            print("Performing cross-validation...")
            cv_scores = cross_val_score(
                model, X_processed, y_processed, cv=5,
                scoring='accuracy' if problem_type == 'classification' else 'r2'
            )
            
            # Get feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_') and self.feature_names:
                feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_') and self.feature_names:
                if len(model.coef_.shape) == 1:
                    feature_importance = dict(zip(self.feature_names, model.coef_))
                else:
                    feature_importance = dict(zip(self.feature_names, 
                                                np.mean(np.abs(model.coef_), axis=0)))
            
            return {
                'metrics': metrics,
                'cv_scores': {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                },
                'feature_importance': feature_importance,
                'model_params': model.get_params(),
                'plots_dir': str(self.plots_dir)
            }
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise 

    def _calculate_clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering performance metrics"""
        metrics = {}
        
        # Skip metrics for DBSCAN when all points are noise (-1)
        if not all(label == -1 for label in labels):
            try:
                metrics['silhouette'] = silhouette_score(X, labels)
            except:
                metrics['silhouette'] = 0
                
            try:
                metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
            except:
                metrics['calinski_harabasz'] = 0
                
            try:
                metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
            except:
                metrics['davies_bouldin'] = 0
        
        metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        return metrics

    def _plot_clusters(self, X: np.ndarray, labels: np.ndarray, model_name: str):
        """Create visualization for clustering results"""
        # Create plots directory
        plots_dir = self.plots_dir / 'clustering'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot first two dimensions
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.title(f'Clustering Results - {model_name}')
        plt.colorbar(scatter)
        plt.savefig(plots_dir / f'{model_name}_clusters.png')
        plt.close()
        
        # Plot cluster sizes
        plt.figure(figsize=(10, 6))
        unique_labels, counts = np.unique(labels, return_counts=True)
        plt.bar([str(l) for l in unique_labels], counts)
        plt.title('Cluster Sizes')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Samples')
        plt.savefig(plots_dir / f'{model_name}_cluster_sizes.png')
        plt.close()

    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Plot elbow curve
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), inertias, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.title('Silhouette Score')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'optimal_clusters.png')
        plt.close()
        
        # Return the number of clusters with highest silhouette score
        return silhouette_scores.index(max(silhouette_scores)) + 2

    def cluster(
        self,
        X: pd.DataFrame,
        model_name: str = 'kmeans',
        n_clusters: Optional[int] = None,
        **model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform clustering on the data"""
        try:
            print("Preprocessing data for clustering...")
            # Scale the features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Find optimal number of clusters if not specified
            if n_clusters is None and model_name == 'kmeans':
                print("Finding optimal number of clusters...")
                n_clusters = self.find_optimal_clusters(X_scaled)
                print(f"Optimal number of clusters: {n_clusters}")
            
            # Initialize clustering model
            model = self.clustering_models[model_name]
            if n_clusters is not None and hasattr(model, 'n_clusters'):
                model_params['n_clusters'] = n_clusters
            
            if model_params:
                model.set_params(**model_params)
            
            # Fit the model
            print(f"Performing {model_name} clustering...")
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            print("Calculating clustering metrics...")
            metrics = self._calculate_clustering_metrics(X_scaled, labels)
            
            # Create visualizations
            print("Creating cluster visualizations...")
            self._plot_clusters(X_scaled, labels, model_name)
            
            return {
                'labels': labels,
                'metrics': metrics,
                'model_params': model.get_params(),
                'plots_dir': str(self.plots_dir / 'clustering')
            }
            
        except Exception as e:
            print(f"Error during clustering: {str(e)}")
            raise 