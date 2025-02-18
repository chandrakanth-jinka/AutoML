from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import shutil

class ReportGenerator:
    def __init__(self):
        self.reports_dir = Path("backend/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        # Create custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        ))
        
    def generate_report(self, 
                       results: Dict[str, Any],
                       dataset_info: Dict[str, Any],
                       model_name: str,
                       problem_type: str,
                       comparison_results: Optional[Dict] = None) -> str:
        """Generate a comprehensive PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.reports_dir / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images directory and copy plots
        images_dir = report_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all plots from the plots directory
        plots_dir = Path(results['plots_dir'])
        if plots_dir.exists():
            for plot_file in plots_dir.glob("*.png"):
                shutil.copy2(plot_file, images_dir / plot_file.name)
        
        report_path = report_dir / "report.pdf"
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Create the story (content) for the PDF
        story = []
        
        # Title
        story.append(Paragraph(f"AutoML Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Dataset Information
        story.append(Paragraph("Dataset Information", self.styles['SectionHeader']))
        dataset_data = [
            ["Total Rows:", str(dataset_info['num_rows'])],
            ["Total Columns:", str(dataset_info['num_columns'])],
            ["Missing Values:", str(sum(dataset_info['missing_values'].values()))]
        ]
        dataset_table = Table(dataset_data, colWidths=[2*inch, 4*inch])
        dataset_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(dataset_table)
        story.append(Spacer(1, 20))
        
        # Model Performance
        story.append(Paragraph(f"Model Performance: {model_name}", self.styles['SectionHeader']))
        story.append(Paragraph(f"Problem Type: {problem_type}", self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Metrics based on problem type
        metrics_data = [["Metric", "Value"]]
        
        if problem_type == 'clustering':
            # Clustering metrics
            metrics_data.extend([
                ["Number of Clusters", str(results['metrics']['n_clusters'])],
                ["Silhouette Score", f"{results['metrics'].get('silhouette', 0):.4f}"],
                ["Calinski-Harabasz Score", f"{results['metrics'].get('calinski_harabasz', 0):.4f}"],
                ["Davies-Bouldin Score", f"{results['metrics'].get('davies_bouldin', 0):.4f}"]
            ])
        elif problem_type == 'classification':
            # Classification metrics
            metrics_data.extend([
                ["Accuracy", f"{results['metrics']['accuracy']:.4f}"],
                ["Macro Precision", f"{results['metrics']['precision_macro']:.4f}"],
                ["Macro Recall", f"{results['metrics']['recall_macro']:.4f}"],
                ["Macro F1", f"{results['metrics']['f1_macro']:.4f}"]
            ])
            if 'roc_auc' in results['metrics']:
                metrics_data.append(["ROC AUC", f"{results['metrics']['roc_auc']:.4f}"])
        else:  # regression
            # Regression metrics
            metrics_data.extend([
                ["R² Score", f"{results['metrics'].get('r2', 0):.4f}"],
                ["RMSE", f"{results['metrics'].get('rmse', 0):.4f}"],
                ["MAE", f"{results['metrics'].get('mae', 0):.4f}"],
                ["Explained Variance", f"{results['metrics'].get('explained_variance', 0):.4f}"]
            ])
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 4*inch])
        metrics_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Visualizations
        story.append(Paragraph("Visualizations", self.styles['SectionHeader']))
        
        # Add visualizations based on problem type
        if problem_type == 'clustering':
            # Add clustering visualizations
            if (images_dir / f"{model_name}_clusters.png").exists():
                story.append(Paragraph("Cluster Visualization", self.styles['Heading3']))
                img = Image(str(images_dir / f"{model_name}_clusters.png"), width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 12))
            
            if (images_dir / f"{model_name}_cluster_sizes.png").exists():
                story.append(Paragraph("Cluster Sizes", self.styles['Heading3']))
                img = Image(str(images_dir / f"{model_name}_cluster_sizes.png"), width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 12))
            
            if model_name == 'kmeans' and (images_dir / "optimal_clusters.png").exists():
                story.append(Paragraph("Optimal Number of Clusters", self.styles['Heading3']))
                img = Image(str(images_dir / "optimal_clusters.png"), width=6*inch, height=4*inch)
                story.append(img)
        else:
            # Add feature importance plot if it exists
            if (images_dir / "feature_importance.png").exists():
                story.append(Paragraph("Feature Importance", self.styles['Heading3']))
                img = Image(str(images_dir / "feature_importance.png"), width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 12))
            
            if problem_type == 'classification':
                # Add classification-specific plots
                if (images_dir / "confusion_matrix.png").exists():
                    story.append(Paragraph("Confusion Matrix", self.styles['Heading3']))
                    img = Image(str(images_dir / "confusion_matrix.png"), width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                
                if (images_dir / "roc_curve.png").exists():
                    story.append(Paragraph("ROC Curve", self.styles['Heading3']))
                    img = Image(str(images_dir / "roc_curve.png"), width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                
                if (images_dir / "precision_recall_curve.png").exists():
                    story.append(Paragraph("Precision-Recall Curve", self.styles['Heading3']))
                    img = Image(str(images_dir / "precision_recall_curve.png"), width=6*inch, height=4*inch)
                    story.append(img)
        
        # Add Model Comparison section if available
        if comparison_results and len(comparison_results) > 1:
            story.append(Paragraph("Model Comparison", self.styles['SectionHeader']))
            
            # Create comparison table
            comparison_data = [["Model", "Performance Score", "Error Rate"]]
            
            for model, res in comparison_results.items():
                metrics = res['metrics']
                if problem_type == 'classification':
                    score = metrics.get('accuracy', 0)
                    error = 1 - score
                elif problem_type == 'regression':
                    score = metrics.get('r2', 0)
                    error = metrics.get('rmse', 0)
                else:  # clustering
                    score = metrics.get('silhouette', 0)
                    error = metrics.get('davies_bouldin', 0)
                
                comparison_data.append([
                    model,
                    f"{score:.4f}",
                    f"{error:.4f}"
                ])
            
            comparison_table = Table(comparison_data, colWidths=[2*inch, 2*inch, 2*inch])
            comparison_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(comparison_table)
            story.append(Spacer(1, 20))
            
            # Add comparison plots
            if (images_dir / "model_comparison.png").exists():
                story.append(Paragraph("Model Performance Comparison", self.styles['Heading3']))
                img = Image(str(images_dir / "model_comparison.png"), width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 12))
        
        # Build the PDF
        doc.build(story)
        
        return str(report_dir) 