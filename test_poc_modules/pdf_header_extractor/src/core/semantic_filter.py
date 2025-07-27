import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from config.settings import (
    EMBEDDING_MODEL, SEMANTIC_SIMILARITY_THRESHOLD, 
    CONTEXT_WINDOW, MAX_PROCESSING_TIME
)
from config.cultural_patterns import CULTURAL_PATTERNS
from src.utils.text_utils import clean_text, extract_sentences
from src.models.embedding_model import EmbeddingModel

import nltk
from nltk.data import find

print(">>> Checking nltk punkt availability...")
try:
    find('tokenizers/punkt/english.pickle')
    print(">>> punkt found.")
except LookupError:
    print(">>> punkt not found, downloading...")
    nltk.download('punkt', quiet=True)
    print(">>> punkt downloaded.")

_old_find = nltk.data.find

def patched_find(resource_name, *args, **kwargs):
    if 'punkt_tab' in resource_name:
        print(f"⚠️ Suppressing missing resource '{resource_name}', substituting with 'tokenizers/punkt/english.pickle'")
        return _old_find('tokenizers/punkt/english.pickle', *args, **kwargs)
    return _old_find(resource_name, *args, **kwargs)

nltk.data.find = patched_find
print(">>> Applied safe punkt_tab monkey patch.")


class SemanticFilter:
    
    def __init__(self, language: str = 'auto', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        self.embedding_model = None
        self._load_embedding_model()
        
        self.similarity_threshold = SEMANTIC_SIMILARITY_THRESHOLD
        self.context_window = CONTEXT_WINDOW
        
        self.heading_patterns = self._load_heading_patterns()
    
    def _load_embedding_model(self) -> None:
        try:
            self.logger.info(f"Initializing embedding model with lazy loading: {EMBEDDING_MODEL}")
            
            self.embedding_model = EmbeddingModel(
                model_name=EMBEDDING_MODEL,
                cache_embeddings=True
            )
            
            self.logger.info("Embedding model initialized successfully with lazy loading")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def _load_heading_patterns(self) -> Dict[str, List[str]]:
        patterns = {
            "introduction": [
                "introduction", "overview", "background", "preface", 
                "foreword", "abstract", "summary", "getting started"
            ],
            "methodology": [
                "methodology", "methods", "approach", "procedure", 
                "technique", "implementation", "design", "framework"
            ],
            "results": [
                "results", "findings", "outcomes", "analysis", 
                "evaluation", "performance", "experiments", "data"
            ],
            "conclusion": [
                "conclusion", "summary", "discussion", "implications", 
                "future work", "recommendations", "final thoughts"
            ],
            "reference": [
                "references", "bibliography", "citations", "sources", 
                "literature", "further reading", "appendix"
            ],
            "chapter": [
                "chapter", "part", "section", "unit", "module", 
                "lesson", "topic", "subject"
            ]
        }
        
        if self.language in CULTURAL_PATTERNS:
            cultural_keywords = CULTURAL_PATTERNS[self.language].get('heading_keywords', [])
            patterns['cultural'] = cultural_keywords
        
        return patterns
    
    def filter_candidates(self, candidates: List, pdf_path: str) -> List:
        if not self.embedding_model or not candidates:
            self.logger.warning("Semantic filtering disabled - model not loaded or no candidates")
            return candidates
        
        self.logger.info(f"Applying semantic filtering to {len(candidates)} candidates")
        
        document_context = self._extract_document_context(pdf_path)
        
        filtered_candidates = []
        
        for candidate in candidates:
            semantic_scores = self._calculate_semantic_scores(
                candidate, document_context, pdf_path
            )
            
            if self._should_keep_candidate(candidate, semantic_scores):
                candidate.features.update({
                    'semantic_scores': semantic_scores,
                    'semantic_verified': True,
                    'context_similarity': semantic_scores.get('context_similarity', 0.0)
                })
                filtered_candidates.append(candidate)
            elif self.debug:
                self.logger.debug(f"Filtered out: '{candidate.text}' - scores: {semantic_scores}")
        
        self.logger.info(f"Semantic filtering: {len(candidates)} -> {len(filtered_candidates)} candidates")
        return filtered_candidates
    
    def _extract_document_context(self, pdf_path: str) -> Dict[str, Any]:
        context = {
            "all_text": "",
            "paragraphs": [],
            "sentences": [],
            "document_type": "unknown",
            "key_terms": set(),
            "page_contexts": {}
        }
        
        try:
            with fitz.open(pdf_path) as doc:
                all_text = []
                
                for page_num in range(min(len(doc), 10)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        all_text.append(page_text)
                        
                        paragraphs = self._extract_paragraphs(page_text)
                        sentences = extract_sentences(page_text)
                        
                        context["paragraphs"].extend(paragraphs)
                        context["sentences"].extend(sentences)
                        context["page_contexts"][page_num + 1] = {
                            "text": page_text,
                            "paragraphs": paragraphs,
                            "sentences": sentences
                        }
                
                context["all_text"] = "\n".join(all_text)
                context["document_type"] = self._detect_document_type(context["all_text"])
                context["key_terms"] = self._extract_key_terms(context["all_text"])
                
        except Exception as e:
            self.logger.warning(f"Failed to extract document context: {e}")
        
        return context
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        cleaned_paragraphs = []
        for para in paragraphs:
            cleaned = clean_text(para)
            if len(cleaned) > 50:
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs
    
    def _detect_document_type(self, text: str) -> str:
        text_lower = text.lower()
        
        academic_indicators = [
            "abstract", "methodology", "references", "citation", 
            "literature review", "hypothesis", "experiment"
        ]
        
        book_indicators = [
            "chapter", "table of contents", "preface", "foreword", 
            "appendix", "index", "bibliography"
        ]
        
        manual_indicators = [
            "installation", "configuration", "troubleshooting", 
            "user guide", "manual", "documentation", "api"
        ]
        
        report_indicators = [
            "executive summary", "findings", "recommendations", 
            "analysis", "quarterly", "annual", "report"
        ]
        
        academic_score = sum(1 for indicator in academic_indicators if indicator in text_lower)
        book_score = sum(1 for indicator in book_indicators if indicator in text_lower)
        manual_score = sum(1 for indicator in manual_indicators if indicator in text_lower)
        report_score = sum(1 for indicator in report_indicators if indicator in text_lower)
        
        scores = {
            "academic": academic_score,
            "book": book_score,
            "manual": manual_score,
            "report": report_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        
        return "general"
    
    def _extract_key_terms(self, text: str) -> set:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        stop_words = {
            'that', 'this', 'with', 'from', 'they', 'been', 'have', 
            'their', 'said', 'each', 'which', 'them', 'than', 'many', 
            'some', 'what', 'time', 'very', 'when', 'much', 'more'
        }
        
        word_freq = defaultdict(int)
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] += 1
        
        key_terms = set()
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]:
            if freq > 2:
                key_terms.add(word)
        
        return key_terms
    
    def _calculate_semantic_scores(self, candidate, document_context: Dict[str, Any], 
                                  pdf_path: str) -> Dict[str, float]:
        scores = {}
        
        candidate_text = candidate.text.strip()
        
        scores['context_similarity'] = self._calculate_context_similarity(
            candidate, document_context, pdf_path
        )
        
        scores['pattern_score'] = self._calculate_pattern_score(candidate_text)
        
        scores['document_coherence'] = self._calculate_document_coherence(
            candidate_text, document_context['document_type']
        )
        
        scores['structural_consistency'] = self._calculate_structural_consistency(
            candidate, document_context
        )
        
        scores['key_term_alignment'] = self._calculate_key_term_alignment(
            candidate_text, document_context['key_terms']
        )
        
        scores['composite_score'] = self._calculate_composite_score(scores)
        
        return scores
    
    def _calculate_context_similarity(self, candidate, document_context: Dict[str, Any], 
                                    pdf_path: str) -> float:
        try:
            candidate_text = candidate.text.strip()
            
            page_context = document_context['page_contexts'].get(candidate.page, {})
            surrounding_paragraphs = page_context.get('paragraphs', [])
            
            if not surrounding_paragraphs:
                return 0.5
            
            context_similarities = []
            for paragraph in surrounding_paragraphs[:self.context_window * 2]:
                if len(paragraph) > 20:
                    similarity = self.embedding_model.compute_similarity(candidate_text, paragraph)
                    context_similarities.append(similarity)
            
            if context_similarities:
                mean_similarity = np.mean(context_similarities)
                if mean_similarity > 0.7:
                    return mean_similarity * 0.5
                return mean_similarity
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"Context similarity calculation failed: {e}")
            return 0.5
    
    def _calculate_pattern_score(self, candidate_text: str) -> float:
        text_lower = candidate_text.lower()
        max_score = 0.0
        
        for pattern_type, patterns in self.heading_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    if text_lower == pattern:
                        max_score = max(max_score, 1.0)
                    elif text_lower.startswith(pattern) or text_lower.endswith(pattern):
                        max_score = max(max_score, 0.8)
                    else:
                        max_score = max(max_score, 0.6)
        
        if re.match(r'^\d+\.', candidate_text) or re.match(r'^[IVX]+\.', candidate_text):
            max_score = max(max_score, 0.7)
        
        return max_score
    
    def _calculate_document_coherence(self, candidate_text: str, document_type: str) -> float:
        text_lower = candidate_text.lower()
        
        coherence_patterns = {
            "academic": ["introduction", "methodology", "results", "discussion", "conclusion"],
            "book": ["chapter", "part", "section", "preface", "appendix"],
            "manual": ["installation", "configuration", "troubleshooting", "guide"],
            "report": ["summary", "analysis", "findings", "recommendations"]
        }
        
        patterns = coherence_patterns.get(document_type, [])
        
        for pattern in patterns:
            if pattern in text_lower:
                return 0.8
        
        return 0.4
    
    def _calculate_structural_consistency(self, candidate, document_context: Dict[str, Any]) -> float:
        
        font_size = candidate.font_size
        page_context = document_context['page_contexts'].get(candidate.page, {})
        
        if font_size > 14:
            return 0.8
        elif font_size > 12:
            return 0.6
        else:
            return 0.4
    
    def _calculate_key_term_alignment(self, candidate_text: str, key_terms: set) -> float:
        candidate_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', candidate_text.lower()))
        
        if not key_terms or not candidate_words:
            return 0.5
        
        overlap = len(candidate_words.intersection(key_terms))
        total_candidate_words = len(candidate_words)
        
        if total_candidate_words == 0:
            return 0.5
        
        alignment_score = overlap / total_candidate_words
        return min(alignment_score * 2, 1.0)
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        weights = {
            'context_similarity': 0.25,
            'pattern_score': 0.30,
            'document_coherence': 0.20,
            'structural_consistency': 0.15,
            'key_term_alignment': 0.10
        }
        
        composite = 0.0
        for score_type, weight in weights.items():
            if score_type in scores:
                composite += scores[score_type] * weight
        
        return composite
    
    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            self.logger.warning(f"Failed to get embedding for text: {e}")
            return np.zeros(384)
    
    def _should_keep_candidate(self, candidate, semantic_scores: Dict[str, float]) -> bool:
        
        composite_score = semantic_scores.get('composite_score', 0.0)
        context_similarity = semantic_scores.get('context_similarity', 0.0)
        pattern_score = semantic_scores.get('pattern_score', 0.0)
        
        base_threshold = 0.2
        
        if (candidate.is_bold or 
            candidate.line_spacing_before > 8 or
            candidate.line_spacing_after > 5):
            base_threshold *= 0.5
        
        if len(candidate.text.split()) <= 3:
            base_threshold *= 0.6
        
        if candidate.text.strip().endswith(':'):
            base_threshold *= 0.4
        
        if hasattr(candidate, 'font_size') and candidate.font_size > 13:
            base_threshold *= 0.7
        
        if composite_score < base_threshold:
            return False
        
        if len(candidate.text.split()) <= 2:
            return composite_score > 0.1
        
        if context_similarity > 0.95 and pattern_score < 0.1 and len(candidate.text.split()) > 5:
            return False
        
        return True
    
    def get_semantic_statistics(self, filtered_candidates: List) -> Dict[str, Any]:
        if not filtered_candidates:
            return {}
        
        scores = []
        pattern_scores = []
        context_scores = []
        
        for candidate in filtered_candidates:
            if hasattr(candidate, 'features') and 'semantic_scores' in candidate.features:
                semantic_scores = candidate.features['semantic_scores']
                scores.append(semantic_scores.get('composite_score', 0.0))
                pattern_scores.append(semantic_scores.get('pattern_score', 0.0))
                context_scores.append(semantic_scores.get('context_similarity', 0.0))
        
        if scores:
            stats = {
                "total_candidates": len(filtered_candidates),
                "avg_composite_score": round(np.mean(scores), 3),
                "avg_pattern_score": round(np.mean(pattern_scores), 3),
                "avg_context_score": round(np.mean(context_scores), 3),
                "score_distribution": {
                    "high (>0.7)": sum(1 for s in scores if s > 0.7),
                    "medium (0.4-0.7)": sum(1 for s in scores if 0.4 <= s <= 0.7),
                    "low (<0.4)": sum(1 for s in scores if s < 0.4)
                }
            }
            
            if self.embedding_model:
                model_stats = self.embedding_model.get_performance_stats()
                stats.update(model_stats)
            
            return stats
        
        return {}
    
    def clear_cache(self) -> None:
        if self.embedding_model:
            self.embedding_model.clear_cache()
        self.logger.debug("All caches cleared")
    
    def preload_model(self) -> bool:
        if self.embedding_model:
            return self.embedding_model.preload_model()
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        if self.embedding_model:
            return self.embedding_model.get_model_info()
        return {"model_loaded": False}