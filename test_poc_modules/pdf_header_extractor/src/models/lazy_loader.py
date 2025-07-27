import logging
import gc
import time
from typing import Optional, Dict, Any
from pathlib import Path
import threading
from sentence_transformers import SentenceTransformer

from config.settings import MODEL_DIR


class LazyModelLoader:
    
    def __init__(self, cache_size_limit: int = 1):
        self.logger = logging.getLogger(__name__)
        self.cache_size_limit = cache_size_limit
        
        self._models: Dict[str, SentenceTransformer] = {}
        self._model_load_times: Dict[str, float] = {}
        self._access_count: Dict[str, int] = {}
        self._lock = threading.RLock()
        
        self.stats = {
            "models_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_load_time": 0.0,
            "memory_clears": 0
        }
        
        self.model_dir = Path(MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"LazyModelLoader initialized with cache limit: {cache_size_limit}")
    
    def load_on_demand(self, model_name: str, device: str = "cpu") -> Optional[SentenceTransformer]:
    
        with self._lock:
            if model_name in self._models:
                self.stats["cache_hits"] += 1
                self._access_count[model_name] += 1
                self.logger.debug(f"Model '{model_name}' loaded from cache")
                return self._models[model_name]
            
            self.stats["cache_misses"] += 1
            self.logger.info(f"Loading model '{model_name}' on device '{device}'...")
            
            try:
                start_time = time.time()
                
                self._manage_cache_size()
                
                model = SentenceTransformer(model_name, device=device)
                model.eval()
                
                load_time = time.time() - start_time
                self._model_load_times[model_name] = load_time
                self._access_count[model_name] = 1
                self.stats["models_loaded"] += 1
                self.stats["total_load_time"] += load_time
                
                self._models[model_name] = model
                
                self.logger.info(f"Model '{model_name}' loaded successfully in {load_time:.2f}s")
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model '{model_name}': {e}")
                return None
    
    def _manage_cache_size(self) -> None:
        if len(self._models) >= self.cache_size_limit:
            lru_model = min(self._access_count.items(), key=lambda x: x[1])[0]
            self.clear_model(lru_model)
            self.logger.info(f"Removed LRU model '{lru_model}' from cache")
    
    def clear_model(self, model_name: str) -> None:
        
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                del self._access_count[model_name]
                if model_name in self._model_load_times:
                    del self._model_load_times[model_name]
                
                gc.collect()
                self.stats["memory_clears"] += 1
                
                self.logger.info(f"Cleared model '{model_name}' from cache")
    
    def clear_all_cache(self) -> None:
        with self._lock:
            model_count = len(self._models)
            
            self._models.clear()
            self._access_count.clear()
            self._model_load_times.clear()
            
            gc.collect()
            self.stats["memory_clears"] += model_count
            
            self.logger.info(f"Cleared all {model_count} models from cache")
    
    def preload_model(self, model_name: str, device: str = "cpu") -> bool:
       
        self.logger.info(f"Preloading model '{model_name}'...")
        model = self.load_on_demand(model_name, device)
        return model is not None
    
    def is_model_loaded(self, model_name: str) -> bool:
       
        with self._lock:
            return model_name in self._models
    
    def get_loaded_models(self) -> list:
        with self._lock:
            return list(self._models.keys())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        
        with self._lock:
            total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
            hit_rate = (self.stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
            
            avg_load_time = (self.stats["total_load_time"] / self.stats["models_loaded"]) if self.stats["models_loaded"] > 0 else 0
            
            return {
                "cache_stats": {
                    "models_currently_loaded": len(self._models),
                    "cache_size_limit": self.cache_size_limit,
                    "loaded_models": list(self._models.keys()),
                    "cache_hit_rate": f"{hit_rate:.1f}%",
                    "cache_hits": self.stats["cache_hits"],
                    "cache_misses": self.stats["cache_misses"],
                    "total_models_loaded": self.stats["models_loaded"],
                    "memory_clears": self.stats["memory_clears"]
                },
                "performance_stats": {
                    "total_load_time": f"{self.stats['total_load_time']:.2f}s",
                    "avg_load_time": f"{avg_load_time:.2f}s",
                    "model_load_times": dict(self._model_load_times),
                    "access_counts": dict(self._access_count)
                }
            }
    
    def optimize_for_inference(self, model_name: str) -> bool:
        
        with self._lock:
            if model_name not in self._models:
                self.logger.warning(f"Model '{model_name}' not loaded, cannot optimize")
                return False
            
            try:
                model = self._models[model_name]
                
                model.eval()
                
                if hasattr(model, '_modules'):
                    for module in model._modules.values():
                        if hasattr(module, 'eval'):
                            module.eval()
                
                self.logger.info(f"Model '{model_name}' optimized for inference")
                return True
                
            except Exception as e:
                self.logger.warning(f"Failed to optimize model '{model_name}': {e}")
                return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        
        memory_info = {
            "loaded_models": len(self._models),
            "estimated_memory_mb": len(self._models) * 80,  # Rough estimate for MiniLM
            "cache_utilization": f"{len(self._models)}/{self.cache_size_limit}"
        }
        
        return memory_info
    
    def warmup(self, model_name: str, sample_texts: list = None) -> None:
        
        model = self.load_on_demand(model_name)
        if model is None:
            return
        
        if sample_texts is None:
            sample_texts = [
                "Introduction",
                "Chapter 1: Overview", 
                "Methodology and Approach",
                "Results and Analysis",
                "Conclusion"
            ]
        
        try:
            self.logger.info(f"Warming up model '{model_name}' with {len(sample_texts)} samples...")
            start_time = time.time()
            
            model.encode(sample_texts, show_progress_bar=False)
            
            warmup_time = time.time() - start_time
            self.logger.info(f"Model warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    def __del__(self):
        try:
            self.clear_all_cache()
        except Exception:
            pass  