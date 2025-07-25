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
from config.cultural_patterns import CULTURAL_PATTERNS


@dataclass
class HeadingCandidate:
    """Represents a potential heading extracted from PDF."""
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
    """Fast candidate generation using font and layout heuristics."""
    
    def __init__(self, language: str = 'auto', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.cultural_patterns = CULTURAL_PATTERNS
        self.document_stats = {}
        
    def generate_candidates(self, pdf_path: str) -> List[HeadingCandidate]:
        """Generate heading candidates from PDF using fast heuristics."""
        self.logger.info(f"Generating candidates for: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        candidates = []
        
        try:
            # First pass: analyze document statistics
            self._analyze_document_stats(doc)
            
            # Second pass: extract candidates
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_candidates = self._extract_page_candidates(page, page_num)
                candidates.extend(page_candidates)
                
        finally:
            doc.close()
            
        # Filter and score candidates
        filtered_candidates = self._filter_candidates(candidates)
        scored_candidates = self._score_candidates(filtered_candidates)
        
        self.logger.info(f"Generated {len(scored_candidates)} candidates")
        return scored_candidates
    
    def _analyze_document_stats(self, doc: fitz.Document) -> None:
        """Analyze document to understand typical font characteristics."""
        font_sizes = []
        font_families = set()
        text_blocks = []
        
        for page_num in range(min(3, len(doc))):  # Sample first 3 pages
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
        
        # Calculate statistics
        self.document_stats = {
            "avg_font_size": np.mean(font_sizes) if font_sizes else 12,
            "median_font_size": np.median(font_sizes) if font_sizes else 12,
            "max_font_size": max(font_sizes) if font_sizes else 12,
            "min_font_size": min(font_sizes) if font_sizes else 12,
            "font_families": font_families,
            "body_text_threshold": np.percentile(font_sizes, 75) if font_sizes else 12
        }
        
        self.logger.debug(f"Document stats: {self.document_stats}")
    
    def _extract_page_candidates(self, page: fitz.Page, page_num: int) -> List[HeadingCandidate]:
        """Extract heading candidates from a single page."""
        candidates = []
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height
        
        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue
                
            # Process each line in the block
            for line_idx, line in enumerate(block["lines"]):
                line_text = ""
                line_bbox = line["bbox"]
                spans = line["spans"]
                
                if not spans:
                    continue
                
                # Combine spans in the line
                dominant_span = max(spans, key=lambda s: (s["size"], len(s["text"])))
                line_text = " ".join([span["text"].strip() for span in spans])
                
                if not self._is_potential_heading_text(line_text):
                    continue
                
                # Calculate features
                features = self._extract_line_features(
                    line, line_text, line_bbox, page.rect.height, page.rect.width, block_idx, line_idx, blocks
                )

                # Create candidate
                candidate = HeadingCandidate(
                    text=line_text.strip(),
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
                    text_length=len(line_text.strip()),
                    features=features
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _extract_line_features(self, line: Dict, text: str, bbox: Tuple,
                           page_height: float, page_width: float,
                           block_idx: int, line_idx: int,
                           blocks: List) -> Dict[str, Any]:

        """Extract detailed features for a line."""
        features = {}
        
        # Spacing analysis
        features["spacing_before"] = self._calculate_spacing_before(
            block_idx, line_idx, blocks
        )
        features["spacing_after"] = self._calculate_spacing_after(
            block_idx, line_idx, blocks
        )
        
        # Text pattern analysis
        features["has_numbering"] = self._has_numbering_pattern(text)
        features["numbering_type"] = self._detect_numbering_type(text)
        features["has_colon"] = text.strip().endswith(':')
        features["all_caps"] = text.isupper()
        features["title_case"] = text.istitle()
        
        # Position features
        features["is_top_of_page"] = bbox[1] / page_height < TITLE_POSITION_THRESHOLD
        center_pos_ratio = ((bbox[0] + bbox[2]) / 2) / page_width
        features["is_centered"] = abs(0.5 - center_pos_ratio) < CENTER_ALIGNMENT_TOLERANCE
        features["indentation"] = bbox[0]
        
        # Language-specific features
        if self.language != 'auto':
            features.update(self._extract_cultural_features(text, self.language))
        
        return features
    
    def _is_potential_heading_text(self, text: str) -> bool:
        """Quick filter for potential heading text."""
        text = text.strip()
        
        if len(text) < MIN_HEADING_LENGTH or len(text) > MAX_HEADING_LENGTH:
            return False
        
        # Skip page numbers, footnotes, headers/footers
        if re.match(r'^\d+$', text):  # Just numbers
            return False
        
        if re.match(r'^[ivxlcdm]+$', text.lower()):  # Roman numerals only
            return False
        
        # Skip common non-heading patterns
        skip_patterns = [
            r'^page \d+',
            r'^\d+/\d+$',  # Page ratios
            r'^www\.',     # URLs
            r'^http',      # URLs
            r'@',          # Email patterns
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, text.lower()):
                return False
        
        return True
    
    def _get_font_weight(self, flags: int) -> str:
        """Extract font weight from flags."""
        if flags & 2**4:  # Bold flag
            return "bold"
        return "normal"
    
    def _determine_alignment(self, bbox: Tuple, page_width: float) -> str:
        """Determine text alignment based on position."""
        left_margin = bbox[0]
        right_margin = page_width - bbox[2]
        center_pos = (bbox[0] + bbox[2]) / 2
        page_center = page_width / 2
        
        # Check if centered
        if abs(center_pos - page_center) / page_width < CENTER_ALIGNMENT_TOLERANCE:
            return "center"
        
        # Check if right-aligned
        if right_margin < left_margin * 0.5:
            return "right"
        
        return "left"
    
    def _calculate_spacing_before(self, block_idx: int, line_idx: int, blocks: List) -> float:
        """Calculate spacing before current line."""
        if block_idx == 0 and line_idx == 0:
            return 0.0
        
        current_bbox = blocks[block_idx]["lines"][line_idx]["bbox"]
        
        # Look for previous line
        if line_idx > 0:
            prev_bbox = blocks[block_idx]["lines"][line_idx - 1]["bbox"]
            return current_bbox[1] - prev_bbox[3]  # Current top - previous bottom
        elif block_idx > 0:
            # Previous block's last line
            prev_block = blocks[block_idx - 1]
            if "lines" in prev_block and prev_block["lines"]:
                prev_bbox = prev_block["lines"][-1]["bbox"]
                return current_bbox[1] - prev_bbox[3]
        
        return 0.0
    
    def _calculate_spacing_after(self, block_idx: int, line_idx: int, blocks: List) -> float:
        """Calculate spacing after current line."""
        current_block = blocks[block_idx]
        current_bbox = current_block["lines"][line_idx]["bbox"]
        
        # Look for next line
        if line_idx < len(current_block["lines"]) - 1:
            next_bbox = current_block["lines"][line_idx + 1]["bbox"]
            return next_bbox[1] - current_bbox[3]
        elif block_idx < len(blocks) - 1:
            # Next block's first line
            for next_block in blocks[block_idx + 1:]:
                if "lines" in next_block and next_block["lines"]:
                    next_bbox = next_block["lines"][0]["bbox"]
                    return next_bbox[1] - current_bbox[3]
        
        return 0.0
    
    def _has_numbering_pattern(self, text: str) -> bool:
        """Check if text has numbering pattern."""
        numbering_patterns = [
            r'^\d+\.',           # 1. 2. 3.
            r'^\d+\.\d+',        # 1.1, 1.2
            r'^[IVX]+\.',        # I. II. III.
            r'^[A-Z]\.',         # A. B. C.
            r'^Chapter \d+',     # Chapter 1
            r'^Section \d+',     # Section 1
            r'^\(\d+\)',         # (1) (2)
        ]
        
        for pattern in numbering_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _detect_numbering_type(self, text: str) -> Optional[str]:
        """Detect the type of numbering used."""
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
        """Extract language-specific features."""
        features = {}
        
        if language in self.cultural_patterns:
            patterns = self.cultural_patterns[language]
            
            # Check for cultural heading styles
            for style in patterns.get('heading_styles', []):
                if style in text:
                    features[f"has_{language}_heading_style"] = True
                    break
            
            # Check for cultural numbering
            for num_pattern in patterns.get('numbering', []):
                if num_pattern in text:
                    features[f"has_{language}_numbering"] = True
                    break
        
        return features
    
    def _filter_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Apply filters to remove unlikely candidates."""
        filtered = []
        
        for candidate in candidates:
            # Font size filter
            if candidate.font_size < self.document_stats["avg_font_size"] * FONT_SIZE_THRESHOLD_RATIO:
                continue
            
            # Skip very short or very long text
            if candidate.text_length < MIN_HEADING_LENGTH or candidate.text_length > MAX_HEADING_LENGTH:
                continue
            
            # Skip if mostly punctuation
            alpha_ratio = sum(c.isalpha() for c in candidate.text) / len(candidate.text)
            if alpha_ratio < 0.3:
                continue
            
            filtered.append(candidate)
        
        self.logger.debug(f"Filtered {len(candidates)} -> {len(filtered)} candidates")
        return filtered
    
    def _score_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Score candidates based on multiple features."""
        for candidate in candidates:
            score = 0.0
            
            # Font size score (0-30 points)
            size_ratio = candidate.font_size / self.document_stats["avg_font_size"]
            score += min(30, size_ratio * 10)
            
            # Bold weight (0-20 points)
            if candidate.is_bold:
                score += 20
            
            # Position score (0-15 points)
            if candidate.features.get("is_top_of_page", False):
                score += 15
            elif candidate.position_ratio < 0.3:  # Upper part of page
                score += 10
            
            # Spacing score (0-15 points)
            if candidate.line_spacing_before > 10:
                score += 8
            if candidate.line_spacing_after > 5:
                score += 7
            
            # Text pattern score (0-20 points)
            if candidate.features.get("has_numbering", False):
                score += 15
            if candidate.features.get("title_case", False):
                score += 5
            if candidate.features.get("has_colon", False):
                score += 5
            
            # Normalize score to 0-1
            candidate.confidence_score = min(1.0, score / 100.0)
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return candidates