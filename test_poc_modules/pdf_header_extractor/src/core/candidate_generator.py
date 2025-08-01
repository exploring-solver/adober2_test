import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import fitz  # PyMuPDF
import numpy as np
from config.settings import (
    FONT_SIZE_THRESHOLD_RATIO, BOLD_WEIGHT_THRESHOLD,
    MIN_HEADING_LENGTH, MAX_HEADING_LENGTH,
    TITLE_POSITION_THRESHOLD, CENTER_ALIGNMENT_TOLERANCE
)
from config.cultural_patterns import CULTURAL_PATTERNS, HEADING_CONFIDENCE_BOOSTERS
from src.utils.text_utils import (
    tokenize_multilingual, 
    enhance_heading_detection_for_cjk,
    is_likely_heading,
    clean_text,
    detect_language
)
from collections import defaultdict

@dataclass
class HeadingCandidate:
    text: str
    page: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float
    font_weight: str
    font_family: str
    is_bold: bool
    is_italic: bool
    alignment: str  # 'left', 'center', 'right'
    position_ratio: float  # 0-1, position on page (0=top)
    line_spacing_before: float
    line_spacing_after: float
    text_length: int
    confidence_score: float = 0.0
    features: Dict[str, Any] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}


class CandidateGenerator:
    
    def __init__(self, language: str = 'auto', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.cultural_patterns = CULTURAL_PATTERNS
        self.document_stats = {}
        self.detected_language = None
        
    def generate_candidates(self, pdf_path: str) -> List[HeadingCandidate]:
        self.logger.info(f"Generating candidates for: {pdf_path}")
        doc = fitz.open(pdf_path)
        all_candidates = []
        
        # First, collect all potential candidates from all pages
        try:
            self._analyze_document_stats(doc)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_candidates = self._extract_page_candidates(page, page_num)
                all_candidates.extend(page_candidates)
                
        finally:
            doc.close()

        # NEW: Identify and exclude running headers/footers
        running_elements = self._identify_running_elements(all_candidates)
        candidates = [cand for cand in all_candidates if cand.text.strip() not in running_elements]

        # Continue with existing filtering and scoring
        filtered_candidates = self._filter_candidates(candidates)
        scored_candidates = self._score_candidates(filtered_candidates)
        
        self.logger.info(f"Generated {len(scored_candidates)} candidates for language: {self.detected_language}")
        return scored_candidates
    
    def _identify_running_elements(self, candidates: List[HeadingCandidate], threshold: int = 2) -> set:
        """Identifies headers or footers based on text that repeats across multiple pages
        in a consistent vertical position."""
        
        text_positions = defaultdict(list)
        page_count = max([c.page for c in candidates] or [1])

        # Collect text and its vertical position ratio for all candidates
        for cand in candidates:
            text_positions[cand.text.strip()].append(cand.position_ratio)

        running_elements = set()
        # For longer documents, require the text to appear on more pages
        min_occurrences = max(threshold, int(page_count * 0.2)) if page_count > 10 else threshold

        for text, positions in text_positions.items():
            # Check if the text appears enough times
            if len(positions) >= min_occurrences:
                # Check if all occurrences are in the top 15% of the page (header)
                is_header = all(p < 0.15 for p in positions)
                # Check if all occurrences are in the bottom 15% of the page (footer)
                is_footer = all(p > 0.85 for p in positions)

                if is_header or is_footer:
                    running_elements.add(text)

        if running_elements:
            self.logger.info(f"Identified and removed running elements: {running_elements}")

        return running_elements
    
    def _analyze_document_stats(self, doc: fitz.Document) -> None:
        font_sizes = []
        font_families = set()
        text_blocks = []
        sample_text = ""
        
        for page_num in range(min(3, len(doc))): 
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
                            font_families.add(span["font"])
                            text_blocks.append({
                                "text": span["text"],
                                "size": span["size"],
                                "flags": span["flags"]
                            })
                            sample_text += span["text"] + " "
        
        if self.language == 'auto':
            self.detected_language = detect_language(sample_text[:1000])
            self.logger.info(f"Detected language: {self.detected_language}")
        else:
            self.detected_language = self.language
        
        self.document_stats = {
            "avg_font_size": np.mean(font_sizes) if font_sizes else 12,
            "median_font_size": np.median(font_sizes) if font_sizes else 12,
            "max_font_size": max(font_sizes) if font_sizes else 12,
            "min_font_size": min(font_sizes) if font_sizes else 12,
            "font_families": font_families,
            "body_text_threshold": np.percentile(font_sizes, 75) if font_sizes else 12,
            "detected_language": self.detected_language,
            "sample_text": sample_text[:500] 
        }
        
        self.logger.debug(f"Document stats: {self.document_stats}") 
    
    def _extract_line_features(self, line: Dict, text: str, bbox: Tuple,
                           page_height: float, page_width: float,
                           block_idx: int, line_idx: int,
                           blocks: List) -> Dict[str, Any]:
        """Extract detailed features for a line with multilingual support."""
        features = {}
        
        features["spacing_before"] = self._calculate_spacing_before(
            block_idx, line_idx, blocks
        )
        features["spacing_after"] = self._calculate_spacing_after(
            block_idx, line_idx, blocks
        )
        
        features["has_numbering"] = self._has_numbering_pattern(text)
        features["numbering_type"] = self._detect_numbering_type(text)
        features["has_colon"] = text.strip().endswith(':')
        features["all_caps"] = text.isupper()
        features["title_case"] = text.istitle()
        
        if self.detected_language in ['japanese', 'chinese']:
            features["is_cjk_chapter"] = self._is_cjk_chapter_heading(text)
            features["is_cjk_section"] = self._is_cjk_section_heading(text)
            features["has_cjk_reject_pattern"] = self._has_cjk_reject_pattern(text)
        
        if self.detected_language:
            tokens = tokenize_multilingual(text, self.detected_language)
            features["token_count"] = len(tokens)
            features["tokens"] = tokens[:10]  
            
            if self.detected_language in ['japanese', 'chinese']:
                cjk_analysis = enhance_heading_detection_for_cjk(text, self.detected_language)
                features["cjk_heading_analysis"] = cjk_analysis
                features["cjk_confidence"] = cjk_analysis.get("confidence", 0.0)
        
        features["is_top_of_page"] = bbox[1] / page_height < TITLE_POSITION_THRESHOLD
        center_pos_ratio = ((bbox[0] + bbox[2]) / 2) / page_width
        features["is_centered"] = abs(0.5 - center_pos_ratio) < CENTER_ALIGNMENT_TOLERANCE
        features["indentation"] = bbox[0]
        
        if self.detected_language and self.detected_language in self.cultural_patterns:
            features.update(self._extract_cultural_features(text, self.detected_language))
        
        features["confidence_boost"] = self._calculate_confidence_boost(text, self.detected_language)
        
        return features
    
    def _is_cjk_chapter_heading(self, text: str) -> bool:
        """Check if text is a CJK chapter heading."""
        chapter_patterns = [
            r'^第[一二三四五六七八九十\d]+章',  # 第一章, 第1章
            r'^第[一二三四五六七八九十\d]+節',  # 第一節, 第1節
            r'^Chapter\s*[一二三四五六七八九十\d]+',  # Chapter 1 (mixed)
        ]
        
        for pattern in chapter_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _is_cjk_section_heading(self, text: str) -> bool:
        """Check if text is a CJK section heading."""
        section_patterns = [
            r'^\d+\.\d+\s+[^\s]',             # 1.1 Title
            r'^[一二三四五六七八九十]+、',        # 一、
            r'^（[一二三四五六七八九十\d]+）',   # （一）
        ]
        
        for pattern in section_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _has_cjk_reject_pattern(self, text: str) -> bool:
        """Check if text matches CJK rejection patterns."""
        reject_patterns = [
            r'^[•·]\s',           # Bullets
            r'^行\d+',            # Table rows
            r'^ヘッダー\d+$',      # Headers
            r'です。$',           # Japanese sentence endings
            r'である。$',         # Japanese sentence endings
            r'した。$',           # Japanese sentence endings
            r'する。$',           # Japanese sentence endings
            r'^表\d+[：:]',       # Table captions (Japanese)
            r'^图\d+[：:]',       # Figure captions (Chinese)
            r'^圖\d+[：:]',       # Figure captions (Traditional Chinese)
            r'^列表\d+',          # List items (Chinese)
            r'^リスト\d+',        # List items (Japanese)
        ]
        
        for pattern in reject_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _extract_page_candidates(self, page: fitz.Page, page_num: int) -> List[HeadingCandidate]:
        """Extract heading candidates from a single page with better text consolidation."""
        candidates = []
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height
        
        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue
            
            for line_idx, line in enumerate(block["lines"]):
                if not line["spans"]:
                    continue
                
                line_text = ""
                dominant_span = None
                max_span_length = 0
                
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        line_text += text + " "
                        if len(text) > max_span_length:
                            max_span_length = len(text)
                            dominant_span = span
                
                line_text = line_text.strip()
                
                if not self._is_potential_heading_text_improved(line_text):
                    continue
                
                line_bbox = line["bbox"]
                
                if not dominant_span:
                    continue
                
                features = self._extract_line_features(
                    line, line_text, line_bbox, page.rect.height, page.rect.width, 
                    block_idx, line_idx, blocks
                )

                candidate = HeadingCandidate(
                    text=line_text,
                    page=page_num + 1,
                    bbox=line_bbox, 
                    font_size=dominant_span["size"],
                    font_weight=self._get_font_weight(dominant_span["flags"]),
                    font_family=dominant_span["font"],
                    is_bold=bool(dominant_span["flags"] & 2**4),
                    is_italic=bool(dominant_span["flags"] & 2**1),
                    alignment=self._determine_alignment(line_bbox, page.rect.width),
                    position_ratio=line_bbox[1] / page_height,
                    line_spacing_before=features["spacing_before"],
                    line_spacing_after=features["spacing_after"],
                    text_length=len(line_text),
                    features=features
                )
                
                candidates.append(candidate)
        
        return candidates

    def _is_potential_heading_text_improved(self, text: str) -> bool:
        text = text.strip()
        
        if len(text) < 3: 
            return False
        if len(text) > 200:  
            return False
        
        if re.match(r'^\d+$', text):
            return False
        if re.match(r'^[\s\.,;:!?\-\(\)\[\]{}]*$', text):
            return False
        
        if re.search(r'(http|www\.|@)', text.lower()):
            return False
        
        if text.endswith('.') and len(text) > 50:
            return False
        if re.match(r'^\d+\.', text) or re.match(r'^[IVX]+\.', text):
            return True
    
        if text.endswith(':'):
            return True
        if (text.istitle() or text.isupper()) and len(text) < 100:
            return True
        
        heading_words = ['chapter', 'section', 'introduction', 'conclusion', 'summary', 'background']
        if any(word in text.lower() for word in heading_words):
            return True
        
        return True  
    
    def _get_font_weight(self, flags: int) -> str:
        """Extract font weight from flags."""
        if flags & 2**4:  
            return "bold"
        return "normal"
    
    def _determine_alignment(self, bbox: Tuple, page_width: float) -> str:
        left_margin = bbox[0]
        right_margin = page_width - bbox[2]
        center_pos = (bbox[0] + bbox[2]) / 2
        page_center = page_width / 2
        
        if abs(center_pos - page_center) / page_width < CENTER_ALIGNMENT_TOLERANCE:
            return "center"
        
        if right_margin < left_margin * 0.5:
            return "right"
        
        return "left"
    
    def _calculate_spacing_before(self, block_idx: int, line_idx: int, blocks: List) -> float:
        if block_idx == 0 and line_idx == 0:
            return 0.0
        
        current_bbox = blocks[block_idx]["lines"][line_idx]["bbox"]
        
        if line_idx > 0:
            prev_bbox = blocks[block_idx]["lines"][line_idx - 1]["bbox"]
            return current_bbox[1] - prev_bbox[3] 
        elif block_idx > 0:
           
            prev_block = blocks[block_idx - 1]
            if "lines" in prev_block and prev_block["lines"]:
                prev_bbox = prev_block["lines"][-1]["bbox"]
                return current_bbox[1] - prev_bbox[3]
        
        return 0.0
    
    def _calculate_spacing_after(self, block_idx: int, line_idx: int, blocks: List) -> float:
        current_block = blocks[block_idx]
        current_bbox = current_block["lines"][line_idx]["bbox"]
        
        
        if line_idx < len(current_block["lines"]) - 1:
            next_bbox = current_block["lines"][line_idx + 1]["bbox"]
            return next_bbox[1] - current_bbox[3]
        elif block_idx < len(blocks) - 1:
           
            for next_block in blocks[block_idx + 1:]:
                if "lines" in next_block and next_block["lines"]:
                    next_bbox = next_block["lines"][0]["bbox"]
                    return next_bbox[1] - current_bbox[3]
        
        return 0.0
    
    def _has_numbering_pattern(self, text: str) -> bool:
        """Check if text has numbering pattern with multilingual support."""
     
        if self.detected_language in self.cultural_patterns:
            patterns = self.cultural_patterns[self.detected_language].get('numbering_patterns', [])
            for pattern in patterns:
                if re.search(pattern, text):
                    return True
        
     
        default_patterns = [
            r'^\d+\.',           # 1. 2. 3.
            r'^\d+\.\d+',        # 1.1, 1.2
            r'^[IVX]+\.',        # I. II. III.
            r'^[A-Z]\.',         # A. B. C.
            r'^Chapter \d+',     # Chapter 1
            r'^Section \d+',     # Section 1
            r'^\(\d+\)',         # (1) (2)
        ]
        
        for pattern in default_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _detect_numbering_type(self, text: str) -> Optional[str]:
        if self.detected_language == 'japanese':
            if re.search(r'^第\d+章', text):
                return "japanese_chapter"
            elif re.search(r'^第[一二三四五六七八九十]+章', text):
                return "japanese_kanji_chapter"
            elif re.search(r'^[一二三四五六七八九十]+、', text):
                return "japanese_kanji_list"
        
        elif self.detected_language == 'chinese':
            if re.search(r'^第\d+章', text):
                return "chinese_chapter"
            elif re.search(r'^第[一二三四五六七八九十]+章', text):
                return "chinese_kanji_chapter"
        
        elif self.detected_language == 'hindi':
            if re.search(r'^अध्याय\s+\d+', text):
                return "hindi_chapter"
            elif re.search(r'^[१२३४५६७८९०]+\.', text):
                return "hindi_devanagari"
        
        elif self.detected_language == 'arabic':
            if re.search(r'^الفصل\s+\w+', text):
                return "arabic_chapter"
            elif re.search(r'^[\u0660-\u0669]+', text):
                return "arabic_indic"
        
        if re.search(r'^\d+\.', text):
            return "decimal"
        elif re.search(r'^[IVX]+\.', text):
            return "roman"
        elif re.search(r'^[A-Z]\.', text):
            return "alpha"
        elif re.search(r'^Chapter', text, re.IGNORECASE):
            return "chapter"
        elif re.search(r'^Section', text, re.IGNORECASE):
            return "section"
        
        return None
    
    def _extract_cultural_features(self, text: str, language: str) -> Dict[str, Any]:
        features = {}
        
        if language in self.cultural_patterns:
            patterns = self.cultural_patterns[language]
            
            heading_styles = patterns.get('heading_styles', [])
            for style in heading_styles:
                if style in text:
                    features[f"has_{language}_heading_style"] = True
                    features[f"heading_style_type"] = style
                    break
            
            heading_keywords = patterns.get('heading_keywords', [])
            for keyword in heading_keywords:
                if keyword in text:
                    features[f"has_{language}_keyword"] = True
                    features[f"keyword_matched"] = keyword
                    break
            
            numbering_patterns = patterns.get('numbering_patterns', [])
            for pattern in numbering_patterns:
                if re.search(pattern, text):
                    features[f"has_{language}_numbering"] = True
                    features[f"numbering_pattern"] = pattern
                    break
            
            if language in ['japanese', 'chinese']:
                features["contains_cjk"] = bool(re.search(r'[\u4e00-\u9fff]', text))
                if language == 'japanese':
                    features["contains_hiragana"] = bool(re.search(r'[\u3041-\u3096]', text))
                    features["contains_katakana"] = bool(re.search(r'[\u30A1-\u30FA]', text))
            
            elif language == 'arabic':
                features["contains_arabic"] = bool(re.search(r'[\u0600-\u06FF]', text))
                features["is_rtl"] = True
            
            elif language == 'hindi':
                features["contains_devanagari"] = bool(re.search(r'[\u0900-\u097F]', text))
        
        return features
    
    def _calculate_confidence_boost(self, text: str, language: str) -> float:
        boost = 0.0
        
        if language in HEADING_CONFIDENCE_BOOSTERS:
            boosters = HEADING_CONFIDENCE_BOOSTERS[language]
            
            for pattern, boost_value in boosters.get('patterns', []):
                if re.search(pattern, text):
                    boost += boost_value
                    break 
            
            for keyword, boost_value in boosters.get('keywords', []):
                if keyword in text:
                    boost += boost_value
                    break 
        
        return min(boost, 1.0)  
    
    def _filter_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        filtered = []
    
        merged_candidates = self._merge_text_fragments(candidates)
        
        for candidate in merged_candidates:
            text = candidate.text.strip()
            
            if len(text) < 3:
                continue
            
            if self._is_text_fragment(text):
                continue
            
            if self._looks_like_body_text(text):
                continue
            
            size_threshold = self.document_stats["avg_font_size"] * 1.1  
            if candidate.font_size < size_threshold:
                if not (candidate.is_bold or text.endswith(':') or 
                    re.match(r'^\d+\.', text) or 
                    any(word in text.lower() for word in ['summary', 'background', 'timeline'])):
                    continue
            
            filtered.append(candidate)
        
        self.logger.debug(f"Filtered {len(candidates)} -> {len(filtered)} candidates")
        return filtered
    
    def _is_text_fragment(self, text: str) -> bool:
        if len(text) < 10 and not text[-1].isalnum() and not text.endswith(':'):
            return True
        
        if text and text[0].islower() and not text.startswith('e.g.'):
            return True
        
        if re.search(r'\b[a-z]{1,2}$', text): 
            return True
        
        return False

    def _looks_like_body_text(self, text: str) -> bool:
        if len(text) > 150:
            return True
        
        if text.count('.') > 1 or text.count('?') > 0:
            return True
        body_starters = ['the ', 'this ', 'that ', 'these ', 'those ', 'it ', 'there ']
        if any(text.lower().startswith(starter) for starter in body_starters):
            return True
        
        return False
    
    def _merge_text_fragments(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Merge text fragments that belong to the same heading with improved logic."""
        if not candidates:
            return candidates
        
        sorted_candidates = sorted(candidates, key=lambda x: (x.page, x.bbox[1], x.bbox[0]))
        
        merged = []
        current_group = [sorted_candidates[0]]
        
        for i in range(1, len(sorted_candidates)):
            current = sorted_candidates[i]
            prev = sorted_candidates[i-1]
            
            # More aggressive merging criteria for multi-line headings
            should_merge = (
                current.page == prev.page and
                abs(current.bbox[1] - prev.bbox[3]) < 15 and  # Vertical proximity (line spacing)
                (
                    # Same horizontal alignment (within margin of error)
                    abs(current.bbox[0] - prev.bbox[0]) < 50 or
                    # Continuation on next line (reasonable horizontal distance)
                    abs(current.bbox[0] - prev.bbox[2]) < 100
                ) and
                # Font size similarity
                abs(current.font_size - prev.font_size) < 2 and
                # Both have similar heading characteristics
                (current.is_bold == prev.is_bold or 
                current.font_size > self.document_stats["avg_font_size"] * 1.1)
            )
            
            if should_merge:
                current_group.append(current)
            else:
                if len(current_group) > 1:
                    merged_candidate = self._merge_candidate_group(current_group)
                    merged.append(merged_candidate)
                else:
                    merged.append(current_group[0])
                
                current_group = [current]
        
        # Handle the last group
        if len(current_group) > 1:
            merged_candidate = self._merge_candidate_group(current_group)
            merged.append(merged_candidate)
        else:
            merged.append(current_group[0])
        
        return merged
    
    def _merge_candidate_group(self, group: List[HeadingCandidate]) -> HeadingCandidate:
        """Merge a group of candidates into a single heading with better text joining."""
        # Join text with appropriate spacing
        text_parts = []
        for candidate in group:
            text = candidate.text.strip()
            if text:
                text_parts.append(text)
        
        # Smart text joining - add space only if needed
        combined_text = ""
        for i, part in enumerate(text_parts):
            if i == 0:
                combined_text = part
            else:
                # Add space if the previous part doesn't end with punctuation
                # and current part doesn't start with punctuation
                if (not combined_text.endswith((':', '-', '—')) and 
                    not part.startswith((':', '-', '—'))):
                    combined_text += " " + part
                else:
                    combined_text += part
        
        # Use the first candidate as base and update with merged properties
        first = group[0]
        last = group[-1]
        
        merged_bbox = (
            first.bbox[0], 
            min(c.bbox[1] for c in group),  
            max(c.bbox[2] for c in group),  
            max(c.bbox[3] for c in group)  
        )
        
        return HeadingCandidate(
            text=combined_text,
            page=first.page,
            bbox=merged_bbox,
            font_size=max(c.font_size for c in group),
            font_weight=first.font_weight,
            font_family=first.font_family,
            is_bold=any(c.is_bold for c in group),
            is_italic=first.is_italic,
            alignment=first.alignment,
            position_ratio=first.position_ratio,
            line_spacing_before=first.line_spacing_before,
            line_spacing_after=last.line_spacing_after,
            text_length=len(combined_text),
            features=first.features.copy()
        )
    def _score_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Score candidates based on multiple features with enhanced CJK scoring."""
        for candidate in candidates:
            score = 0.0
            
           
            size_ratio = candidate.font_size / self.document_stats["avg_font_size"]
            score += min(30, size_ratio * 10)
            
           
            if candidate.is_bold:
                score += 20
            
            if candidate.features.get("is_top_of_page", False):
                score += 15
            elif candidate.position_ratio < 0.3: 
                score += 10
            
            if candidate.line_spacing_before > 10:
                score += 8
            if candidate.line_spacing_after > 5:
                score += 7
            
            if candidate.features.get("has_numbering", False):
                score += 15
            if candidate.features.get("title_case", False):
                score += 5
            if candidate.features.get("has_colon", False):
                score += 5
            
            if self.detected_language in ['japanese', 'chinese']:
                if candidate.font_size < self.document_stats["avg_font_size"] * 1.2:
                    if not (candidate.features.get("is_cjk_chapter", False) or 
                           candidate.features.get("is_cjk_section", False)):
                        score *= 0.3  
                if candidate.features.get("is_cjk_chapter", False):
                    score += 40  
                elif candidate.features.get("is_cjk_section", False):
                    score += 25 
                
               
                cjk_confidence = candidate.features.get("cjk_confidence", 0.0)
                score += cjk_confidence * 15
                
                if candidate.features.get("contains_cjk", False):
                    score += 10
            
            if self.detected_language:
             
                boost = candidate.features.get("confidence_boost", 0.0)
                score += boost * 20 
                
                if candidate.features.get(f"has_{self.detected_language}_heading_style", False):
                    score += 15
                if candidate.features.get(f"has_{self.detected_language}_numbering", False):
                    score += 10
            
            linguistic_confidence = candidate.features.get("linguistic_analysis", {}).get("confidence", 0.0)
            score += linguistic_confidence * 10
            
            candidate.confidence_score = min(1.0, score / 100.0)
        
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        self.logger.debug(f"Top candidate scores: {[c.confidence_score for c in candidates[:5]]}")
        return candidates
    
    def get_generation_stats(self) -> Dict[str, Any]:
        return {
            "detected_language": self.detected_language,
            "document_stats": self.document_stats,
            "cultural_patterns_used": self.detected_language in self.cultural_patterns,
            "tokenization_available": self.detected_language in ['japanese', 'chinese']
        }