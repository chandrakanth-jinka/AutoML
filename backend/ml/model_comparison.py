from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ModelComparison:
    def __init__(self):
        self.plots_dir = Path("backend/plots/comparison")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models and their metrics"""
        comparison_data = []
        
        # Get the problem type from the first result
        first_result = next(iter(results_dict.values()))
        is_clustering = 'n_clusters' in first_result['metrics']
        
        for model_name, results in results_dict.items():
            metrics = results['metrics']
            model_data = {'Model': model_name}
            
            if is_clustering:
                model_data.update({
                    'Number of Clusters': metrics['n_clusters'],
                    'Silhouette Score': metrics.get('silhouette', 0),
                    'Calinski-Harabasz Score': metrics.get('calinski_harabasz', 0),
                    'Davies-Bouldin Score': metrics.get('davies_bouldin', 0)
                })
            else:
                # Existing classification/regression comparison code
                model_data.update(metrics)
                if 'cv_scores' in results:
                    model_data['CV Mean Score'] = results['cv_scores']['mean']
                    model_data['CV Std'] = results['cv_scores']['std']
            
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        self._plot_comparison(comparison_df, is_clustering)
        return comparison_df
    
    def _plot_comparison(self, df: pd.DataFrame, is_clustering: bool):
        """Create comparison visualizations"""
        # Metrics comparison plot
        plt.figure(figsize=(12, 6))
        metrics_to_plot = [col for col in df.columns if col not in ['Model', 'CV Std']]
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='Model', y=metric)
            plt.title(f'{metric} Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{metric.lower().replace(" ", "_")}_comparison.png')
            plt.close()
        
        # CV scores with error bars for non-clustering models
        if not is_clustering and 'CV Mean Score' in df.columns and 'CV Std' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.errorbar(df['Model'], df['CV Mean Score'], 
                        yerr=df['CV Std'], fmt='o', capsize=5)
            plt.title('Cross-validation Scores Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'cv_scores_comparison.png')
            plt.close()
        
        # Clustering specific plots
        if is_clustering:
            plt.figure(figsize=(10, 6))
            plt.bar(df['Model'], df['Number of Clusters'])
            plt.title('Number of Clusters Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'n_clusters_comparison.png')
            plt.close()
            
            # Plot silhouette scores
            plt.figure(figsize=(10, 6))
            plt.bar(df['Model'], df['Silhouette Score'])
            plt.title('Silhouette Score Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'silhouette_score_comparison.png')
            plt.close() 