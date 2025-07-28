import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import fitz
import pdfplumber
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from src.core.candidate_generator import CandidateGenerator
from src.core.semantic_filter import SemanticFilter
from src.core.hierarchy_assigner import HierarchyAssigner
from src.core.output_formatter import OutputFormatter
from src.utils.validation import validate_pdf, detect_language
from src.utils.text_utils import clean_text, normalize_whitespace
from config.settings import (
    MAX_PROCESSING_TIME, MAX_FILE_SIZE_MB, 
    OUTPUT_DIR, INCLUDE_DEBUG_INFO
)


class PDFProcessor:
    
    def __init__(self, language: str = 'auto', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        self.candidate_generator = CandidateGenerator(language=language, debug=debug)
        self.semantic_filter = SemanticFilter(language=language, debug=debug) if not self._is_fast_mode() else None
        self.hierarchy_assigner = HierarchyAssigner(language=language, debug=debug)
        self.output_formatter = OutputFormatter(debug=debug)
        
        self.stats = {
            "start_time": None,
            "end_time": None,
            "processing_stages": [],
            "warnings": [],
            "document_info": {}
        }
    
    def process(self, pdf_path: str, timeout: Optional[int] = None, 
                include_metadata: Optional[bool] = None) -> Dict[str, Any]:
        self.stats["start_time"] = time.time()
        timeout = timeout or MAX_PROCESSING_TIME
        
        if include_metadata is None:
            include_metadata = self._should_include_metadata()
        
        if include_metadata:
            self.logger.info(f"Starting PDF processing with full metadata and accessibility support: {pdf_path}")
        else:
            self.logger.info(f"Starting PDF processing in simple mode: {pdf_path}")
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._process_internal, pdf_path, include_metadata)
                result = future.result(timeout=timeout)
                
        except FutureTimeoutError:
            self.logger.error(f"Processing timeout after {timeout}s")
            raise TimeoutError(f"PDF processing exceeded {timeout} second limit")
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise
        
        self.stats["end_time"] = time.time()
        self.stats["processing_time"] = self.stats["end_time"] - self.stats["start_time"]
        
        self.logger.info(f"Processing completed in {self.stats['processing_time']:.2f}s")
        return result
    
    def process_for_round1a(self, pdf_path: str) -> Dict[str, Any]:
        self.logger.info("Processing for Round 1A format (no accessibility metadata)")
        
        result = self.process(pdf_path, include_metadata=False)
        
        return {
            "title": result.get("title", "Document"),
            "outline": result.get("outline", [])
        }
    
    def _process_internal(self, pdf_path: str, include_metadata: bool = False) -> Dict[str, Any]:
        
        self._add_stage("pdf_validation")
        document_info = self._analyze_pdf(pdf_path)
        
        self._add_stage("structure_detection")
        structured_headings = self._extract_structured_headings(pdf_path)
        
        headings = [] # Initialize headings list
        hierarchy_tree = None

        if structured_headings:
            self.logger.info("Found structured PDF tags, using native extraction")
            headings = structured_headings
            hierarchy_tree = self._build_simple_tree(headings) if include_metadata else None
        else:
            self._add_stage("candidate_generation")
            candidates = self.candidate_generator.generate_candidates(pdf_path)
            
            if not candidates:
                self.logger.warning("No heading candidates found")
                return self._create_empty_result(document_info, include_metadata)
            
            if self.semantic_filter and not self._is_fast_mode():
                self._add_stage("semantic_filtering")
                filtered_candidates = self.semantic_filter.filter_candidates(
                    candidates, pdf_path
                )
            else:
                filtered_candidates = candidates
            
            self._add_stage("hierarchy_assignment")
            headings = self.hierarchy_assigner.assign_hierarchy(filtered_candidates)
            
            if include_metadata:
                hierarchy_tree = self.hierarchy_assigner.generate_hierarchy_tree(
                    [self._dict_to_node(h) for h in headings]
                )
            else:
                hierarchy_tree = None
        
        self._add_stage("output_formatting")
        
        processing_stats = None
        if include_metadata:
            processing_stats = {
                **self.stats,
                "hierarchy_stats": self.hierarchy_assigner.get_hierarchy_statistics(
                    [self._dict_to_node(h) for h in headings]
                ) if headings else {}
            }
        
        result = self.output_formatter.format_results(
            headings=headings,
            document_info={**document_info, **self.stats},
            hierarchy_tree=hierarchy_tree,
            processing_stats=processing_stats if (include_metadata and INCLUDE_DEBUG_INFO) else None,
            include_metadata=include_metadata
        )
        
        # --- MODIFICATION STARTS HERE ---
        # Determine the most appropriate title and override if necessary
        chosen_title = self._select_document_title(document_info, headings)
        if result.get("title") != chosen_title:
            result["title"] = chosen_title
            self.logger.info(f"Final document title selected: '{chosen_title}'")
        # --- MODIFICATION ENDS HERE ---

        if include_metadata and "accessibility" in result:
            acc_summary = result["accessibility"]["compliance_summary"]
            self.logger.info(f"Accessibility Score: {acc_summary['accessibility_score']:.1f}/100")
            self.logger.info(f"Compliance - WCAG: {'✓' if acc_summary['wcag_2_1_aa'] else '✗'}, "
                           f"PDF/UA: {'✓' if acc_summary['pdf_ua'] else '✗'}, "
                           f"Section 508: {'✓' if acc_summary['section_508'] else '✗'}")
        elif not include_metadata:
            self.logger.info(f"Simple format generated with {len(headings)} headings")
        
        return result
    
    def _analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        
        if not validate_pdf(pdf_path):
            raise ValueError(f"Invalid PDF file: {pdf_path}")
        
        file_size = os.path.getsize(pdf_path)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            self.logger.warning(f"Large file size: {file_size / (1024*1024):.1f}MB")
        
        document_info = {
            "filename": Path(pdf_path).name,
            "file_path": str(pdf_path),
            "file_size": file_size,
            "processing_method": "hybrid"
        }
        
        try:
            with fitz.open(pdf_path) as doc:
                metadata = doc.metadata
                document_info.update({
                    "total_pages": len(doc),
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", ""),
                })
                
                if self.language == 'auto':
                    detected_lang = self._detect_document_language(doc)
                    document_info["language"] = detected_lang
                    self.language = detected_lang
                else:
                    document_info["language"] = self.language
                
                if self._should_include_metadata():
                    structure_info = self._analyze_document_structure(doc)
                    document_info.update(structure_info)
                
        except Exception as e:
            self.logger.warning(f"Failed to extract PDF metadata: {e}")
            document_info.update({
                "total_pages": 0,
                "language": self.language,
            })
        
        return document_info
    
    def _extract_structured_headings(self, pdf_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract structured headings from PDF TOC/outline with validation against visible content.
        This method now validates that TOC entries actually exist as visible text in the document.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if not doc.is_pdf or not hasattr(doc, 'get_toc'):
                    return None
                
                toc = doc.get_toc()
                if not toc:
                    return None
                
                self.logger.info(f"Found structured TOC with {len(toc)} entries - validating against visible content")
                
                # Extract all visible text from document for validation
                visible_text_by_page = {}
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text().strip()
                    visible_text_by_page[page_num + 1] = page_text.lower()
                
                structured_headings = []
                validated_count = 0
                
                for i, (level, title, page_num) in enumerate(toc):
                    try:
                        title_clean = clean_text(title).strip()
                        
                        # Skip empty or very short titles
                        if not title_clean or len(title_clean) < 2:
                            self.logger.debug(f"Skipping empty/short TOC entry: '{title}'")
                            continue
                        
                        # Validate that this heading actually exists in the visible content
                        page_text = visible_text_by_page.get(page_num, "")
                        title_variations = [
                            title_clean.lower(),
                            title_clean.lower().replace(' ', ''),  # Remove spaces
                            title_clean.lower().replace('-', ' '),  # Replace hyphens
                            title_clean.lower().replace('_', ' '),  # Replace underscores
                        ]
                        
                        # Check if any variation of the title exists in the page text
                        found_in_visible_text = any(variation in page_text for variation in title_variations)
                        
                        if not found_in_visible_text:
                            self.logger.debug(f"TOC entry '{title_clean}' not found in visible text on page {page_num} - skipping")
                            continue
                        
                        # Additional validation: try to find the text location on the page
                        try:
                            page = doc.load_page(page_num - 1)
                            text_instances = page.search_for(title_clean)
                            
                            # If we can't find it with exact search, try partial matches
                            if not text_instances:
                                # Try searching for significant words (longer than 3 chars)
                                words = [w for w in title_clean.split() if len(w) > 3]
                                if words:
                                    # Search for the longest word
                                    longest_word = max(words, key=len)
                                    text_instances = page.search_for(longest_word)
                            
                            if text_instances:
                                bbox = text_instances[0]
                            else:
                                # If still not found, this might be a phantom TOC entry
                                self.logger.debug(f"Could not locate TOC entry '{title_clean}' on page - might be phantom entry")
                                bbox = [0, 0, 100, 20]  # Default bbox, but mark with low confidence
                        except Exception as e:
                            self.logger.debug(f"Error locating TOC entry '{title_clean}': {e}")
                            bbox = [0, 0, 100, 20]
                        
                        heading = {
                            "text": title_clean,
                            "level": max(1, min(level, 6)),
                            "page": page_num,
                            "bbox": list(bbox),
                            "font_info": {
                                "size": 14,
                                "weight": "bold",
                                "family": "unknown"
                            },
                            "confidence": 0.9 if found_in_visible_text else 0.3,  # Lower confidence for unverified entries
                            "features": {
                                "source": "pdf_structure",
                                "toc_index": i,
                                "validated_against_content": found_in_visible_text
                            }
                        }
                        
                        structured_headings.append(heading)
                        validated_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process TOC entry '{title}': {e}")
                        continue
                
                if structured_headings:
                    self.logger.info(f"Validated {validated_count}/{len(toc)} TOC entries against visible content")
                    return structured_headings
                else:
                    self.logger.info("No valid TOC entries found after content validation - falling back to text analysis")
                    return None
                    
        except Exception as e:
            self.logger.debug(f"Structured extraction failed: {e}")
            return None
    
    def _detect_document_language(self, doc: fitz.Document) -> str:
        
        sample_text = ""
        for page_num in range(min(3, len(doc))):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            sample_text += page_text[:1000]
        
        if len(sample_text.strip()) < 100:
            return 'en'
        
        detected_language = detect_language(sample_text)
        self.logger.debug(f"Detected language: {detected_language}")
        
        return detected_language
    
    def _analyze_document_structure(self, doc: fitz.Document) -> Dict[str, Any]:
        
        structure_info = {
            "has_images": False,
            "has_tables": False,
            "is_multi_column": False,
            "avg_line_height": 0,
            "font_analysis": {},
            "layout_complexity": "simple"
        }
        
        try:
            font_sizes = []
            font_families = set()
            line_heights = []
            has_images = False
            
            for page_num in range(min(3, len(doc))):
                page = doc.load_page(page_num)
                
                if page.get_images():
                    has_images = True
                
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        if len(line["spans"]) > 0:
                            for span in line["spans"]:
                                font_sizes.append(span["size"])
                                font_families.add(span["font"])
                            
                            bbox = line["bbox"]
                            line_heights.append(bbox[3] - bbox[1])
                
                if self._detect_multi_column_layout(page):
                    structure_info["is_multi_column"] = True
            
            if font_sizes:
                structure_info["font_analysis"] = {
                    "unique_sizes": len(set(font_sizes)),
                    "size_range": [min(font_sizes), max(font_sizes)],
                    "avg_size": sum(font_sizes) / len(font_sizes),
                    "font_families": list(font_families)
                }
            
            if line_heights:
                structure_info["avg_line_height"] = sum(line_heights) / len(line_heights)
            
            structure_info["has_images"] = has_images
            
            complexity_score = 0
            if structure_info["is_multi_column"]: complexity_score += 2
            if has_images: complexity_score += 1
            if len(font_families) > 3: complexity_score += 1
            if structure_info["font_analysis"].get("unique_sizes", 0) > 5: complexity_score += 1
            
            if complexity_score >= 4:
                structure_info["layout_complexity"] = "complex"
            elif complexity_score >= 2:
                structure_info["layout_complexity"] = "moderate"
                
        except Exception as e:
            self.logger.warning(f"Structure analysis failed: {e}")
        
        return structure_info
    
    def _detect_multi_column_layout(self, page: fitz.Page) -> bool:
        
        try:
            blocks = page.get_text("dict")["blocks"]
            text_blocks = [b for b in blocks if "lines" in b]
            
            if len(text_blocks) < 4:
                return False
            
            left_blocks = []
            right_blocks = []
            page_width = page.rect.width
            middle = page_width / 2
            
            for block in text_blocks:
                bbox = block["bbox"]
                block_center = (bbox[0] + bbox[2]) / 2
                
                if block_center < middle * 0.8:
                    left_blocks.append(block)
                elif block_center > middle * 1.2:
                    right_blocks.append(block)
            
            return len(left_blocks) >= 2 and len(right_blocks) >= 2
            
        except Exception:
            return False
    
    def _build_simple_tree(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        tree = {}
        stack = []
        
        for heading in headings:
            level = heading["level"]
            text = heading["text"]
            
            while len(stack) >= level:
                stack.pop()
            
            node = {
                "level": level,
                "page": heading["page"],
                "children": {}
            }
            
            if not stack:
                tree[text] = node
                stack.append((text, node))
            else:
                parent_name, parent_node = stack[-1]
                parent_node["children"][text] = node
                stack.append((text, node))
        
        return tree
    
    def _create_empty_result(self, document_info: Dict[str, Any], include_metadata: bool = False) -> Dict[str, Any]:
        
        self.stats["warnings"].append("No headings detected in document")
        
        return self.output_formatter.format_results(
            headings=[],
            document_info={**document_info, **self.stats},
            hierarchy_tree={} if include_metadata else None,
            processing_stats=self.stats if (include_metadata and INCLUDE_DEBUG_INFO) else None,
            include_metadata=include_metadata
        )
    
    def _dict_to_node(self, heading_dict: Dict[str, Any]):
        from src.core.hierarchy_assigner import HierarchyNode
        
        return HierarchyNode(
            text=heading_dict["text"],
            level=heading_dict["level"],
            page=heading_dict["page"],
            bbox=tuple(heading_dict["bbox"]),
            font_size=heading_dict["font_info"]["size"],
            confidence=heading_dict.get("confidence", 0.0)
        )
    
    def _add_stage(self, stage_name: str) -> None:
        
        stage_info = {
            "name": stage_name,
            "timestamp": time.time(),
            "duration": None
        }
        
        if self.stats["processing_stages"]:
            prev_stage = self.stats["processing_stages"][-1]
            prev_stage["duration"] = stage_info["timestamp"] - prev_stage["timestamp"]
        
        self.stats["processing_stages"].append(stage_info)
        self.logger.debug(f"Starting stage: {stage_name}")
    
    def _is_fast_mode(self) -> bool:
        return os.getenv("FAST_MODE", "false").lower() == "true"
    
    def _should_include_metadata(self) -> bool:
        return os.getenv("INCLUDE_METADATA", "false").lower() == "true"
    
    def _select_document_title(self, document_info: Dict[str, Any], headings: List[Dict[str, Any]]) -> str:
        """
        Intelligently compares the PDF's metadata title and the first detected heading to select the most appropriate one.
        Enhanced with comprehensive logging and smart comparison heuristics.
        """
        self.logger.info("=== SMART DOCUMENT TITLE SELECTION ===")
        
        metadata_title = document_info.get("title", "").strip()
        first_heading_text = headings[0]["text"].strip() if headings else ""
        
        self.logger.info(f"PDF Metadata Title: '{metadata_title}'")
        self.logger.info(f"First Heading Text: '{first_heading_text}'")
        
        # Enhanced generic phrases list
        generic_phrases = [
            "untitled", "document", "document1", "new document", "microsoft word",
            "final", "draft", "report", "presentation", "summary", "pdf",
            "docx", "doc", "page 1", "header", "footer", "temp", "copy of",
            "blank", "title", "heading", "chapter 1", "section 1"
        ]

        def is_generic_or_empty(text: str) -> bool:
            """Enhanced generic detection with detailed logging"""
            if not text or len(text) < 3:
                self.logger.debug(f"'{text}' is generic: too short (length: {len(text)})")
                return True
            
            text_lower = text.lower()
            for phrase in generic_phrases:
                if phrase in text_lower:
                    self.logger.debug(f"'{text}' is generic: contains '{phrase}'")
                    return True
            
            # Additional checks for generic patterns
            if text_lower.endswith(('.pdf', '.docx', '.doc', '.ppt', '.pptx')):
                self.logger.debug(f"'{text}' is generic: ends with file extension")
                return True
            
            if text_lower.startswith('microsoft word - '):
                self.logger.debug(f"'{text}' is generic: Microsoft Word prefix")
                return True
            
            self.logger.debug(f"'{text}' passed generic check")
            return False

        def score_title_quality(text: str) -> float:
            """Score title quality from 0.0 to 1.0"""
            if not text:
                return 0.0
            
            score = 0.5  # Base score
            text_clean = text.strip()
            
            # Length scoring (optimal range: 10-60 characters)
            length = len(text_clean)
            if 10 <= length <= 60:
                score += 0.2
            elif 5 <= length <= 80:
                score += 0.1
            elif length < 5 or length > 100:
                score -= 0.2
            
            # Content indicators
            if any(c.isalpha() for c in text_clean):  # Contains letters
                score += 0.1
            
            if text_clean.count(' ') >= 1:  # Multi-word title
                score += 0.1
            
            if text_clean and text_clean[0].isupper() and not text_clean.isupper():  # Proper case
                score += 0.1
            
            if ':' in text_clean and text_clean.count(':') == 1:  # Subtitle
                score += 0.1
            
            # Penalties
            if is_generic_or_empty(text_clean):
                score -= 0.4
            
            if text_clean.startswith(('Chapter', 'Section', 'Part')):
                score -= 0.2
            
            if text_clean.isupper():  # All caps
                score -= 0.1
            
            # Domain-specific bonuses
            academic_indicators = ['analysis', 'study', 'research', 'investigation', 'report']
            if any(indicator in text_clean.lower() for indicator in academic_indicators):
                score += 0.1
            
            return max(0.0, min(1.0, score))

        # Heuristic 1: If there are no headings, fall back to metadata title or a default
        if not first_heading_text:
            self.logger.info("CASE 1: No headings found")
            if metadata_title:
                self.logger.info(f"✓ DECISION: Using metadata title: '{metadata_title}'")
                return metadata_title
            else:
                fallback_title = "Document"
                self.logger.info(f"⚠ DECISION: Using fallback title: '{fallback_title}'")
                return fallback_title

        # Heuristic 2: If metadata title is generic or empty, prefer the first heading
        metadata_is_generic = is_generic_or_empty(metadata_title)
        heading_is_generic = is_generic_or_empty(first_heading_text)
        
        self.logger.info(f"Quality assessment:")
        self.logger.info(f"  Metadata title is generic: {metadata_is_generic}")
        self.logger.info(f"  First heading is generic: {heading_is_generic}")
        
        if metadata_is_generic and not heading_is_generic:
            self.logger.info("CASE 2: Metadata title is generic, first heading is good")
            self.logger.info(f"✓ DECISION: Using first heading: '{first_heading_text}'")
            return first_heading_text
        
        if heading_is_generic and not metadata_is_generic:
            self.logger.info("CASE 3: First heading is generic, metadata title is good")
            self.logger.info(f"✓ DECISION: Using metadata title: '{metadata_title}'")
            return metadata_title

        # Heuristic 3: Both are present and need detailed comparison
        if not metadata_is_generic and not heading_is_generic:
            self.logger.info("CASE 4: Both titles are good quality - detailed comparison")
            
            # Score both titles
            metadata_score = score_title_quality(metadata_title)
            heading_score = score_title_quality(first_heading_text)
            
            self.logger.info(f"Quality scores:")
            self.logger.info(f"  Metadata: {metadata_score:.3f} - '{metadata_title}'")
            self.logger.info(f"  Heading:  {heading_score:.3f} - '{first_heading_text}'")
            
            # Length comparison for additional context
            metadata_len = len(metadata_title)
            heading_len = len(first_heading_text)
            length_ratio = heading_len / metadata_len if metadata_len > 0 else float('inf')
            
            self.logger.info(f"Length analysis:")
            self.logger.info(f"  Metadata length: {metadata_len}")
            self.logger.info(f"  Heading length: {heading_len}")
            self.logger.info(f"  Length ratio (heading/metadata): {length_ratio:.2f}")
            
            # Decision logic
            score_diff = heading_score - metadata_score
            
            if heading_score > metadata_score + 0.1:  # Heading significantly better
                self.logger.info(f"✓ DECISION: First heading significantly better (score diff: +{score_diff:.3f}): '{first_heading_text}'")
                return first_heading_text
            elif metadata_score > heading_score + 0.1:  # Metadata significantly better
                self.logger.info(f"✓ DECISION: Metadata title significantly better (score diff: {score_diff:.3f}): '{metadata_title}'")
                return metadata_title
            else:
                # Scores are close - use length heuristic
                self.logger.info(f"Scores are close (diff: {score_diff:.3f}) - using length heuristic")
                
                if length_ratio > 1.5:  # First heading significantly longer
                    self.logger.info(f"✓ DECISION: First heading much longer and descriptive: '{first_heading_text}'")
                    return first_heading_text
                elif length_ratio < 0.67:  # Metadata significantly longer
                    self.logger.info(f"✓ DECISION: Metadata title much longer and descriptive: '{metadata_title}'")
                    return metadata_title
                else:
                    # Default to first heading (content-derived)
                    self.logger.info(f"✓ DECISION: Defaulting to first heading (content-derived): '{first_heading_text}'")
                    return first_heading_text
        
        # Heuristic 4: Both are generic - choose the less generic one
        else:
            self.logger.info("CASE 5: Both titles are generic - choosing less generic option")
            
            metadata_score = score_title_quality(metadata_title)
            heading_score = score_title_quality(first_heading_text)
            
            self.logger.info(f"Generic quality scores:")
            self.logger.info(f"  Metadata: {metadata_score:.3f}")
            self.logger.info(f"  Heading: {heading_score:.3f}")
            
            if heading_score >= metadata_score:
                self.logger.info(f"✓ DECISION: First heading less generic: '{first_heading_text}'")
                return first_heading_text
            else:
                self.logger.info(f"✓ DECISION: Metadata title less generic: '{metadata_title}'")
                return metadata_title

    def save_output(self, result: Dict[str, Any], output_path: Optional[str] = None, 
           formats: Optional[List[str]] = None, 
           auto_filename: bool = True) -> Dict[str, str]:
        
        from config.settings import JSON_OUTPUT_DIR, OUTPUT_DIR
        
        if formats is None:
            formats = ["json"]
        
        if output_path is None or auto_filename:
            if "title" in result:
                filename = result["title"]
            elif "document_info" in result:
                filename = result["document_info"].get("filename", "outline")
            else:
                filename = "outline"
            
            safe_name = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_name = f"{safe_name}_{timestamp}"
        else:
            final_name = Path(output_path).stem
        
        output_files = {}
        
        try:
            for format_type in formats:
                if format_type == "json":
                    output_dir = JSON_OUTPUT_DIR
                    extension = ".json"
                elif format_type == "pdf_ua_xml":
                    output_dir = OUTPUT_DIR
                    extension = "_accessibility.xml"
                else:
                    output_dir = OUTPUT_DIR
                    extension = f".{format_type}"
                
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"{final_name}{extension}"
                
                if format_type == "json":
                    self.output_formatter.save_json(result, output_file)
                elif format_type == "pdf_ua_xml":
                    headings = result.get("outline", result.get("headings", []))
                    self.output_formatter.save_pdf_ua_xml(headings, output_file)
                
                output_files[format_type] = str(output_file)
                self.logger.info(f"Saved {format_type.upper()} output to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save output: {e}")
            raise
        
        return output_files

    def save_output_to_custom_path(self, result: Dict[str, Any], custom_path: str, 
                                formats: Optional[List[str]] = None) -> Dict[str, str]:
        
        if formats is None:
            formats = ["json"]
        
        custom_path = Path(custom_path)
        output_files = {}
        
        try:
            custom_path.parent.mkdir(parents=True, exist_ok=True)
            
            if len(formats) == 1:
                format_type = formats[0]
                
                if format_type == "json":
                    self.output_formatter.save_json(result, custom_path)
                elif format_type == "csv":
                    self.output_formatter.save_csv(result, custom_path)
                elif format_type == "xml":
                    self.output_formatter.save_xml(result, custom_path)
                elif format_type == "markdown":
                    self.output_formatter.save_markdown(result, custom_path)
                elif format_type == "html":
                    self.output_formatter.save_html_outline(result, custom_path)
                elif format_type == "pdf_ua_xml":
                    headings = result.get("outline", result.get("headings", []))
                    self.output_formatter.save_pdf_ua_xml(headings, custom_path)
                
                output_files[format_type] = str(custom_path)
            else:
                base_path = custom_path.with_suffix('')
                
                for format_type in formats:
                    if format_type == "pdf_ua_xml":
                        format_path = base_path.with_suffix('_accessibility.xml')
                    else:
                        format_path = base_path.with_suffix(f'.{format_type}')
                    
                    if format_type == "json":
                        self.output_formatter.save_json(result, format_path)
                    elif format_type == "csv":
                        self.output_formatter.save_csv(result, format_path)
                    elif format_type == "xml":
                        self.output_formatter.save_xml(result, format_path)
                    elif format_type == "markdown":
                        self.output_formatter.save_markdown(result, format_path)
                    elif format_type == "html":
                        self.output_formatter.save_html_outline(result, format_path)
                    elif format_type == "pdf_ua_xml":
                        headings = result.get("outline", result.get("headings", []))
                        self.output_formatter.save_pdf_ua_xml(headings, format_path)
                    
                    output_files[format_type] = str(format_path)
            
            self.logger.info(f"Saved output to custom path: {custom_path.parent}")
            
        except Exception as e:
            self.logger.error(f"Failed to save to custom path: {e}")
            raise
        
        return output_files

    def process_batch(self, pdf_paths: List[str], 
                     output_dir: Optional[str] = None,
                     max_workers: int = 2,
                     include_accessibility: bool = False,
                     include_metadata: bool = False) -> Dict[str, Any]:
        
        output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        failed = {}
        
        self.logger.info(f"Starting batch processing of {len(pdf_paths)} files")
        if include_metadata:
            self.logger.info("Full metadata will be included")
        if include_accessibility:
            self.logger.info("Accessibility XML files will be generated")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.process, pdf_path, None, include_metadata): pdf_path 
                for pdf_path in pdf_paths
            }
            
            for future in future_to_path:
                pdf_path = future_to_path[future]
                try:
                    result = future.result(timeout=MAX_PROCESSING_TIME)
                    results[pdf_path] = result
                    
                    formats = ["json"]
                    if include_accessibility:
                        formats.append("pdf_ua_xml")
                    
                    output_file = output_dir / f"{Path(pdf_path).stem}_headings"
                    self.save_output_to_custom_path(result, output_file, formats)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {pdf_path}: {e}")
                    failed[pdf_path] = str(e)
        
        accessibility_summary = {}
        if include_accessibility and include_metadata and results:
            total_score = 0
            compliant_count = 0
            
            for pdf_path, result in results.items():
                if "accessibility" in result:
                    acc_data = result["accessibility"]["compliance_summary"]
                    total_score += acc_data["accessibility_score"]
                    if acc_data["wcag_2_1_aa"]:
                        compliant_count += 1
            
            accessibility_summary = {
                "average_accessibility_score": total_score / len(results),
                "wcag_compliant_documents": compliant_count,
                "compliance_rate": (compliant_count / len(results)) * 100
            }
        
        batch_summary = {
            "total_files": len(pdf_paths),
            "successful": len(results),
            "failed": len(failed),
            "success_rate": len(results) / len(pdf_paths) * 100,
            "failed_files": failed,
            "output_directory": str(output_dir),
            "metadata_included": include_metadata,
            "accessibility_included": include_accessibility,
            "accessibility_summary": accessibility_summary
        }
        
        summary_file = output_dir / "batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Batch processing complete: {len(results)}/{len(pdf_paths)} successful")
        if accessibility_summary:
            self.logger.info(f"Average accessibility score: {accessibility_summary['average_accessibility_score']:.1f}/100")
            self.logger.info(f"WCAG compliant: {accessibility_summary['compliance_rate']:.1f}%")
        
        return {
            "results": results,
            "summary": batch_summary
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        
        if not self.stats["processing_stages"]:
            return {}
        
        total_time = self.stats.get("processing_time", 0)
        stage_breakdown = {}
        
        for stage in self.stats["processing_stages"]:
            if stage["duration"] is not None:
                stage_breakdown[stage["name"]] = {
                    "duration": round(stage["duration"], 3),
                    "percentage": round((stage["duration"] / total_time) * 100, 1) if total_time > 0 else 0
                }
        
        return {
            "total_processing_time": total_time,
            "stage_breakdown": stage_breakdown,
            "warnings": self.stats["warnings"],
            "document_analysis": self.stats.get("document_info", {})
        }
    
    def clear_caches(self) -> None:
        
        try:
            if self.semantic_filter:
                self.semantic_filter.clear_cache()
            
            if hasattr(self.candidate_generator, 'clear_cache'):
                self.candidate_generator.clear_cache()
            
            if hasattr(self.hierarchy_assigner, 'clear_cache'):
                self.hierarchy_assigner.clear_cache()
            
            self.logger.info("All component caches cleared")
            
        except Exception as e:
            self.logger.warning(f"Failed to clear some caches: {e}")
            
    def get_accessibility_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        
        if "accessibility" not in result:
            return {"accessibility_available": False}
        
        acc_data = result["accessibility"]
        summary = {
            "accessibility_available": True,
            "compliance_summary": acc_data["compliance_summary"],
            "total_issues": len(acc_data["metadata"]["issues"]),
            "recommendations_count": len(acc_data["metadata"]["recommendations"]),
            "structure_xml_available": acc_data["structure_xml_available"]
        }
        
        return summary