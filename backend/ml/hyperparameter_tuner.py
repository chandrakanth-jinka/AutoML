from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

class HyperparameterTuner:
    def __init__(self):
        self.param_grids = {
            'classification': {
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'logistic_regression': {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [1000]
                },
                'svm': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                },
                'xgboost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'regression': {
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'linear_regression': {},  # Linear regression doesn't have hyperparameters to tune
                'svr': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'epsilon': [0.1, 0.2, 0.3]
                },
                'xgboost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'clustering': {
                'kmeans': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                    'init': ['k-means++', 'random'],
                    'n_init': [10, 20, 30]
                },
                'dbscan': {
                    'eps': [0.1, 0.2, 0.3, 0.5, 0.8],
                    'min_samples': [3, 5, 7, 10],
                    'metric': ['euclidean', 'manhattan']
                },
                'hierarchical': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                    'affinity': ['euclidean', 'manhattan'],
                    'linkage': ['ward', 'complete', 'average']
                }
            }
        }

    def tune_hyperparameters(
        self,
        X,
        y=None,
        model_name: str = None,
        problem_type: str = None,
        cv: int = 5,
        n_iter: int = 10,
        method: str = 'random'
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using either GridSearchCV or RandomizedSearchCV
        
        Args:
            X: Features
            y: Target (optional, not needed for clustering)
            model_name: Name of the model to tune
            problem_type: 'classification' or 'regression' or 'clustering'
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            method: 'grid' or 'random' search
        """
        if problem_type == 'clustering':
            return self._tune_clustering(X, model_name)
            
        if y is None:
            raise ValueError("Target variable 'y' is required for classification/regression")
            
        # Get parameter grid
        param_grid = self.param_grids[problem_type].get(model_name)
        if not param_grid:
            return {"message": f"No hyperparameters to tune for {model_name}"}

        # Initialize base model
        base_model = self._get_base_model(model_name, problem_type)
        
        # Initialize search
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                n_jobs=-1,
                verbose=1,
                scoring='accuracy' if problem_type == 'classification' else 'r2'
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                n_jobs=-1,
                verbose=1,
                scoring='accuracy' if problem_type == 'classification' else 'r2'
            )

        # Fit search
        search.fit(X, y)
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'best_model': search.best_estimator_
        }

    def _get_base_model(self, model_name: str, problem_type: str):
        """Get the base model for hyperparameter tuning"""
        models = {
            'classification': {
                'random_forest': RandomForestClassifier(),
                'logistic_regression': LogisticRegression(),
                'svm': SVC(probability=True),
                'xgboost': xgb.XGBClassifier()
            },
            'regression': {
                'random_forest': RandomForestRegressor(),
                'linear_regression': LinearRegression(),
                'svr': SVR(),
                'xgboost': xgb.XGBRegressor()
            },
            'clustering': {
                'kmeans': KMeans(),
                'dbscan': DBSCAN(),
                'hierarchical': AgglomerativeClustering()
            }
        }
        return models[problem_type][model_name]

    def _tune_clustering(self, X, model_name: str):
        """Tune clustering model hyperparameters"""
        param_grid = self.param_grids['clustering'].get(model_name)
        if not param_grid:
            return {"message": f"No hyperparameters to tune for {model_name}"}

        best_score = -float('inf')
        best_params = None
        best_model = None

        # For each parameter combination
        for params in ParameterGrid(param_grid):
            model = self._get_base_model(model_name, 'clustering')
            model.set_params(**params)
            
            # Fit and evaluate
            labels = model.fit_predict(X)
            
            # Calculate silhouette score
            try:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
            except:
                continue

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model
        } 