import os
from ml.data_loader import DataLoader
from ml.data_processor import PreprocessingService
from ml.model_trainer import ModelTrainer
from ml.model_comparison import ModelComparison
from ml.hyperparameter_tuner import HyperparameterTuner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ml.model_persistence import ModelPersistence
from ml.report_generator import ReportGenerator
from datetime import datetime

# Add this after the imports
AVAILABLE_MODELS = {
    'classification': ['random_forest', 'logistic_regression', 'svm', 'xgboost'],
    'regression': ['random_forest', 'linear_regression', 'svr', 'xgboost']
}

def analyze_and_visualize_data(df: pd.DataFrame, summary: dict):
    """Analyze data and create visualizations"""
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
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                
                # Create safe filename by replacing invalid characters
                safe_filename = "".join(c if c.isalnum() else "_" for c in col)
                plt.savefig(plots_dir / f'{safe_filename}_distribution.png')
                plt.close()
                
                # Basic statistics
                print(f"  - Mean: {df[col].mean():.2f}")
                print(f"  - Std: {df[col].std():.2f}")
                print(f"  - Min: {df[col].min():.2f}")
                print(f"  - Max: {df[col].max():.2f}")
            except Exception as e:
                print(f"  Warning: Could not create distribution plot for {col}: {str(e)}")
        
        # Value counts for categorical columns
        if col in summary['categorical_columns']:
            print("  - Top categories:")
            for val, count in df[col].value_counts().head().items():
                print(f"    {val}: {count}")
    
    # Correlation matrix for numeric columns
    if summary['numeric_columns']:
        try:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df[summary['numeric_columns']].corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(plots_dir / 'correlation_matrix.png')
            plt.close()
        except Exception as e:
            print(f"\nWarning: Could not create correlation matrix: {str(e)}")
    
    print(f"\nData visualizations saved in: {plots_dir}")
    return plots_dir

def suggest_problem_type(column_name: str, df: pd.DataFrame, summary: dict) -> str:
    """Suggest whether the target is for classification or regression"""
    if column_name in summary['categorical_columns']:
        unique_count = df[column_name].nunique()
        if unique_count < 10:  # Arbitrary threshold
            return 'classification'
    if column_name in summary['numeric_columns']:
        return 'regression'
    return 'classification'  # Default to classification

def get_user_input(df: pd.DataFrame, summary: dict):
    """Get user inputs for model configuration"""
    target_column = None
    n_clusters = None
    
    # First ask for problem type
    print("\nSelect problem type:")
    print("1. Classification")
    print("2. Regression")
    print("3. Clustering")
    
    while True:
        try:
            problem_type_idx = int(input("\nEnter problem type number: "))
            if problem_type_idx == 1:
                problem_type = 'classification'
                break
            elif problem_type_idx == 2:
                problem_type = 'regression'
                break
            elif problem_type_idx == 3:
                problem_type = 'clustering'
                break
            else:
                print("Invalid input. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # If not clustering, get target column
    if problem_type != 'clustering':
        print("\nAvailable columns:")
        for i, col in enumerate(summary['columns'], 1):
            print(f"{i}. {col}")
        
        while True:
            try:
                col_idx = int(input("\nEnter target column number: ")) - 1
                target_column = summary['columns'][col_idx]
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid column number.")
    
    # Select appropriate model based on problem type
    if problem_type == 'clustering':
        models = ['kmeans', 'dbscan', 'hierarchical']
        print(f"\nAvailable clustering models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                model_idx = int(input("\nSelect clustering model number: ")) - 1
                model_name = models[model_idx]
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid model number.")
        
        # For k-means and hierarchical, optionally specify number of clusters
        if model_name in ['kmeans', 'hierarchical']:
            try:
                n_clusters = int(input("\nEnter number of clusters (press Enter for automatic): ") or 0)
                if n_clusters <= 0:
                    n_clusters = None
            except ValueError:
                n_clusters = None
    else:
        print(f"\nAvailable models for {problem_type}:")
        for i, model in enumerate(AVAILABLE_MODELS[problem_type], 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                model_idx = int(input("\nSelect model number: ")) - 1
                model_name = AVAILABLE_MODELS[problem_type][model_idx]
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid model number.")
    
    return target_column, problem_type, model_name, n_clusters

def run_automl():
    """Main AutoML function"""
    print("Welcome to AutoML!")
    
    # Get CSV file path
    while True:
        file_path = input("\nEnter the path to your CSV file: ")
        if os.path.exists(file_path) and file_path.endswith('.csv'):
            break
        print("File not found or not a CSV file. Please enter a valid path.")
    
    try:
        # Load and analyze data
        data_loader = DataLoader()
        df, summary = data_loader.load_file(file_path)
        
        # Analyze and visualize data
        plots_dir = analyze_and_visualize_data(df, summary)
        
        # Preprocess data first
        print("\n=== Data Preprocessing ===")
        preprocessor = PreprocessingService()
        df_processed, preprocess_info = preprocessor.preprocess_data(df)
        
        print("\nPreprocessing Summary:")
        for key, value in preprocess_info.items():
            print(f"{key}: {value}")
        
        # Get user inputs
        target_column, problem_type, model_name, n_clusters = get_user_input(df, summary)
        
        # Initialize model trainer
        model_trainer = ModelTrainer()
        
        if problem_type == 'clustering':
            # For clustering, we use all features
            X = df_processed
            
            # Perform clustering
            results = model_trainer.cluster(
                X,
                model_name=model_name,
                n_clusters=n_clusters
            )
            
            # Display clustering results
            print("\n=== Clustering Results ===")
            print("\nMetrics:")
            for metric, value in results['metrics'].items():
                print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
            
            print(f"\nCluster labels saved and plots created in: {results['plots_dir']}")
            
            # Ask if user wants to try other clustering models
            try_others = input("\nWould you like to try other clustering models for comparison? (y/n): ").lower()
            if try_others == 'y':
                results_dict = {model_name: results}
                
                for other_model in ['kmeans', 'dbscan', 'hierarchical']:
                    if other_model != model_name:
                        print(f"\n=== Training {other_model.upper()} Clustering ===")
                        other_results = model_trainer.cluster(
                            X,
                            model_name=other_model,
                            n_clusters=n_clusters
                        )
                        results_dict[other_model] = other_results
                
                # Compare models
                print("\n=== Model Comparison ===")
                model_comparison = ModelComparison()
                comparison_df = model_comparison.compare_models(results_dict)
                
                print("\nClustering Model Comparison Summary:")
                print(comparison_df.to_string())
                print(f"\nComparison plots saved in: {model_comparison.plots_dir}")
            
            # Ask if user wants to tune hyperparameters
            tune_params = input("\nWould you like to tune clustering parameters? (y/n): ").lower()
            if tune_params == 'y':
                print("\nTuning parameters...")
                tuner = HyperparameterTuner()
                tuning_results = tuner.tune_hyperparameters(
                    X=X,
                    model_name=model_name,
                    problem_type='clustering'
                )
                
                if 'best_params' in tuning_results:
                    print("\nBest parameters found:")
                    for param, value in tuning_results['best_params'].items():
                        print(f"{param}: {value}")
                    print(f"\nBest silhouette score: {tuning_results['best_score']:.4f}")
                    
                    # Use the tuned model
                    model = tuning_results['best_model']
            
            # Save clustering model
            save_model = input("\nWould you like to save this clustering model? (y/n): ").lower()
            if save_model == 'y':
                model_persistence = ModelPersistence()
                metadata = {
                    'model_name': model_name,
                    'problem_type': 'clustering',
                    'metrics': results['metrics'],
                    'feature_names': list(X.columns),
                    'training_date': datetime.now().isoformat()
                }
                saved_path = model_persistence.save_model(
                    model_trainer.trained_model,
                    metadata,
                    model_name
                )
                print(f"\nModel saved to: {saved_path}")
                
                # Ask for predictions
                predict_now = input("\nWould you like to make predictions with this model? (y/n): ").lower()
                if predict_now == 'y':
                    input_method = input("\nChoose input method:\n1. Enter values manually\n2. Upload test CSV file\nEnter choice (1/2): ")
                    
                    if input_method == '1':
                        # Manual input
                        print("\nEnter values for each feature:")
                        input_values = {}
                        for feature in X.columns:
                            while True:
                                try:
                                    value = float(input(f"{feature}: "))
                                    input_values[feature] = value
                                    break
                                except ValueError:
                                    print("Please enter a valid number")
                        
                        # Create DataFrame and preprocess
                        input_df = pd.DataFrame([input_values])
                        input_processed, _ = preprocessor.preprocess_data(input_df)
                        
                        # Make prediction
                        prediction = model.predict(input_processed)
                        
                        print("\nPrediction Result:")
                        if problem_type == 'classification':
                            print(f"Predicted class: {prediction[0]}")
                        elif problem_type == 'regression':
                            print(f"Predicted value: {prediction[0]:.4f}")
                        else:  # clustering
                            print(f"Assigned cluster: {prediction[0]}")
                            
                    elif input_method == '2':
                        # File input
                        test_file = input("\nEnter path to test CSV file: ")
                        if os.path.exists(test_file):
                            # Load and preprocess test data
                            test_df, _ = data_loader.load_file(test_file)
                            
                            # Validate columns
                            missing_cols = set(X.columns) - set(test_df.columns)
                            if missing_cols:
                                print(f"\nError: Test file missing required columns: {missing_cols}")
                            else:
                                # Preprocess test data
                                test_processed, _ = preprocessor.preprocess_data(test_df)
                                
                                # Make predictions
                                predictions = model.predict(test_processed)
                                
                                # Add predictions to test data
                                test_df['predictions'] = predictions
                                
                                # Save results
                                results_file = f"predictions_{os.path.basename(test_file)}"
                                results_path = os.path.join("backend/predictions", results_file)
                                os.makedirs("backend/predictions", exist_ok=True)
                                test_df.to_csv(results_path, index=False)
                                
                                print("\nPrediction Results:")
                                print(f"Predictions saved to: {results_path}")
                                print("\nFirst 5 predictions:")
                                for i, pred in enumerate(predictions[:5], 1):
                                    print(f"Row {i}: {pred:.4f}")
                        else:
                            print("Error: Test file not found")
                    else:
                        print("Invalid choice")
            
            # Generate report
            generate_report = input("\nWould you like to generate a detailed report? (y/n): ").lower()
            if generate_report == 'y':
                report_generator = ReportGenerator()
                report_path = report_generator.generate_report(
                    results,
                    summary,
                    model_name,
                    problem_type
                )
                print(f"\nReport generated: {report_path}")
        
        else:
            # For classification/regression
            X = df_processed.drop(columns=[target_column])
            y = df[target_column]
            
            # Train model
            print(f"\n=== Training {model_name.upper()} Model ===")
            results = model_trainer.train(
                X, y,
                problem_type=problem_type,
                model_name=model_name
            )
            
            # Display results
            print("\n=== Model Results ===")
            if problem_type == 'classification':
                print("\nOverall Metrics:")
                print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
                print(f"Macro-averaged Metrics:")
                print(f"Precision: {results['metrics']['precision_macro']:.4f}")
                print(f"Recall: {results['metrics']['recall_macro']:.4f}")
                print(f"F1-score: {results['metrics']['f1_macro']:.4f}")
                
                print("\nWeighted-averaged Metrics:")
                print(f"Precision: {results['metrics']['precision_weighted']:.4f}")
                print(f"Recall: {results['metrics']['recall_weighted']:.4f}")
                print(f"F1-score: {results['metrics']['f1_weighted']:.4f}")
                
                if 'roc_auc' in results['metrics']:
                    print(f"\nROC AUC: {results['metrics']['roc_auc']:.4f}")
                elif 'roc_auc_macro' in results['metrics']:
                    print(f"\nROC AUC (macro): {results['metrics']['roc_auc_macro']:.4f}")
                
                print("\nPer-class Metrics:")
                for metric, value in results['metrics'].items():
                    if metric.startswith('precision_class_'):
                        class_name = metric.replace('precision_class_', '')
                        print(f"\nClass: {class_name}")
                        print(f"Precision: {value:.4f}")
                        print(f"Recall: {results['metrics'][f'recall_class_{class_name}']:.4f}")
                        print(f"F1-score: {results['metrics'][f'f1_class_{class_name}']:.4f}")
                
            else:  # regression
                print("\nMetrics:")
                print(f"R² Score: {results['metrics']['r2']:.4f}")
                print(f"Mean Squared Error: {results['metrics']['mse']:.4f}")
                print(f"Root Mean Squared Error: {results['metrics']['rmse']:.4f}")
                print(f"Mean Absolute Error: {results['metrics']['mae']:.4f}")
                print(f"Explained Variance: {results['metrics']['explained_variance']:.4f}")
            
            print("\nCross-validation Scores:")
            print(f"Mean: {results['cv_scores']['mean']:.4f}")
            print(f"Std: {results['cv_scores']['std']:.4f}")
            
            if results['feature_importance']:
                print("\nTop 5 Important Features:")
                sorted_features = dict(sorted(
                    results['feature_importance'].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5])
                for feature, importance in sorted_features.items():
                    print(f"{feature}: {importance:.4f}")
            
            print(f"\nPlots saved in: {results['plots_dir']}")
            print("\nGenerated plots include:")
            if problem_type == 'classification':
                print("- Confusion Matrix")
                print("- ROC Curve")
                print("- Precision-Recall Curve")
            print("- Feature Importance Plot")
            
            # Ask if user wants to try other models
            try_others = input("\nWould you like to try other models for comparison? (y/n): ").lower()
            if try_others == 'y':
                results_dict = {model_name: results}  # Store first model's results
                
                # Try other available models
                available_models = AVAILABLE_MODELS[problem_type]
                
                for other_model in available_models:
                    if other_model != model_name:
                        print(f"\n=== Training {other_model.upper()} Model ===")
                        other_results = model_trainer.train(
                            X, y,
                            problem_type=problem_type,
                            model_name=other_model
                        )
                        results_dict[other_model] = other_results
                
                # Compare models
                print("\n=== Model Comparison ===")
                model_comparison = ModelComparison()
                comparison_df = model_comparison.compare_models(results_dict)
                
                print("\nModel Comparison Summary:")
                print(comparison_df.to_string())
                print(f"\nComparison plots saved in: {model_comparison.plots_dir}")
            
            # Ask if user wants to tune hyperparameters
            tune_params = input("\nWould you like to tune model hyperparameters? (y/n): ").lower()
            if tune_params == 'y':
                print("\nTuning hyperparameters...")
                tuner = HyperparameterTuner()
                tuning_results = tuner.tune_hyperparameters(
                    X, y,
                    model_name=model_name,
                    problem_type=problem_type
                )
                
                if 'best_params' in tuning_results:
                    print("\nBest parameters found:")
                    for param, value in tuning_results['best_params'].items():
                        print(f"{param}: {value}")
                    print(f"\nBest cross-validation score: {tuning_results['best_score']:.4f}")
                    
                    # Use the tuned model
                    model = tuning_results['best_model']
            
            # Save model if desired
            save_model = input("\nWould you like to save this model? (y/n): ").lower()
            if save_model == 'y':
                model_persistence = ModelPersistence()
                metadata = {
                    'model_name': model_name,
                    'problem_type': problem_type,
                    'metrics': results['metrics'],
                    'feature_names': list(X.columns),
                    'training_date': datetime.now().isoformat()
                }
                saved_path = model_persistence.save_model(
                    model_trainer.trained_model,
                    metadata,
                    model_name
                )
                print(f"\nModel saved to: {saved_path}")
                
                # Ask for predictions
                predict_now = input("\nWould you like to make predictions with this model? (y/n): ").lower()
                if predict_now == 'y':
                    input_method = input("\nChoose input method:\n1. Enter values manually\n2. Upload test CSV file\nEnter choice (1/2): ")
                    
                    if input_method == '1':
                        # Manual input
                        print("\nEnter values for each feature:")
                        input_values = {}
                        for feature in X.columns:
                            while True:
                                try:
                                    value = float(input(f"{feature}: "))
                                    input_values[feature] = value
                                    break
                                except ValueError:
                                    print("Please enter a valid number")
                        
                        # Create DataFrame and preprocess
                        input_df = pd.DataFrame([input_values])
                        input_processed, _ = preprocessor.preprocess_data(input_df)
                        
                        # Make prediction
                        prediction = model.predict(input_processed)
                        
                        print("\nPrediction Result:")
                        if problem_type == 'classification':
                            print(f"Predicted class: {prediction[0]}")
                        elif problem_type == 'regression':
                            print(f"Predicted value: {prediction[0]:.4f}")
                        else:  # clustering
                            print(f"Assigned cluster: {prediction[0]}")
                            
                    elif input_method == '2':
                        # File input
                        test_file = input("\nEnter path to test CSV file: ")
                        if os.path.exists(test_file):
                            # Load and preprocess test data
                            test_df, _ = data_loader.load_file(test_file)
                            
                            # Validate columns
                            missing_cols = set(X.columns) - set(test_df.columns)
                            if missing_cols:
                                print(f"\nError: Test file missing required columns: {missing_cols}")
                            else:
                                # Preprocess test data
                                test_processed, _ = preprocessor.preprocess_data(test_df)
                                
                                # Make predictions
                                predictions = model.predict(test_processed)
                                
                                # Add predictions to test data
                                test_df['predictions'] = predictions
                                
                                # Save results
                                results_file = f"predictions_{os.path.basename(test_file)}"
                                results_path = os.path.join("backend/predictions", results_file)
                                os.makedirs("backend/predictions", exist_ok=True)
                                test_df.to_csv(results_path, index=False)
                                
                                print("\nPrediction Results:")
                                print(f"Predictions saved to: {results_path}")
                                print("\nFirst 5 predictions:")
                                for i, pred in enumerate(predictions[:5], 1):
                                    print(f"Row {i}: {pred:.4f}")
                        else:
                            print("Error: Test file not found")
                    else:
                        print("Invalid choice")
            
            # Generate report
            generate_report = input("\nWould you like to generate a detailed report? (y/n): ").lower()
            if generate_report == 'y':
                report_generator = ReportGenerator()
                report_path = report_generator.generate_report(
                    results,
                    summary,
                    model_name,
                    problem_type
                )
                print(f"\nReport generated: {report_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    run_automl() 