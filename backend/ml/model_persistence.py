import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import datetime

class ModelPersistence:
    def __init__(self):
        self.models_dir = Path("backend/saved_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, model: Any, metadata: Dict, model_name: str, results_dict: Optional[Dict] = None) -> str:
        """Save model if it's the best performing one"""
        # Check if this is the best model
        if results_dict and len(results_dict) > 1:
            current_score = self._get_model_score(metadata['metrics'], metadata['problem_type'])
            for other_results in results_dict.values():
                other_score = self._get_model_score(other_results['metrics'], metadata['problem_type'])
                if other_score > current_score:
                    return None  # Don't save if not the best model
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Add performance ranking to metadata
        if results_dict:
            metadata['performance_ranking'] = self._get_performance_ranking(
                results_dict, metadata['problem_type']
            )
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return str(model_dir)
    
    def _get_model_score(self, metrics: Dict, problem_type: str) -> float:
        """Get a comparable score based on problem type"""
        if problem_type == 'classification':
            return metrics.get('accuracy', 0)
        elif problem_type == 'regression':
            return metrics.get('r2', 0)
        else:  # clustering
            return metrics.get('silhouette', 0)
            
    def _get_performance_ranking(self, results_dict: Dict, problem_type: str) -> Dict:
        """Rank models by performance"""
        scores = {}
        for model_name, results in results_dict.items():
            scores[model_name] = self._get_model_score(results['metrics'], problem_type)
            
        # Sort models by score
        ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {
            'rankings': [{'model': m, 'score': s} for m, s in ranked_models],
            'best_model': ranked_models[0][0],
            'best_score': ranked_models[0][1]
        }
    
    def load_model(self, model_dir: str) -> Tuple[Any, Dict]:
        """Load model and its metadata"""
        model_path = Path(model_dir) / "model.joblib"
        metadata_path = Path(model_dir) / "metadata.json"
        
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Model or metadata file not found")
        
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return model, metadata 