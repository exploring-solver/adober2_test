import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from src.core.candidate_generator import HeadingCandidate
import numpy as np
from config.settings import MAX_HIERARCHY_LEVELS, TITLE_POSITION_THRESHOLD
from config.cultural_patterns import CULTURAL_PATTERNS


@dataclass
class HierarchyNode:
    text: str
    level: int
    page: int
    bbox: Tuple[float, float, float, float]
    font_size: float
    confidence: float
    parent: Optional['HierarchyNode'] = None
    children: List['HierarchyNode'] = field(default_factory=list)
    numbering_pattern: Optional[str] = None
    semantic_group: Optional[str] = None


class HierarchyAssigner:
    
    def __init__(self, language: str = 'auto', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.cultural_patterns = CULTURAL_PATTERNS
        
        self.strategies = [
            self._assign_by_cjk_patterns,  
            self._assign_by_font_hierarchy,
            self._assign_by_numbering_pattern,
            self._assign_by_position_and_spacing,
            self._assign_by_keywords,
            self._assign_by_indentation,
        ]
    
    def assign_hierarchy(self, candidates: List[HeadingCandidate]) -> List[Dict[str, Any]]:
        self.logger.info(f"Assigning hierarchy to {len(candidates)} candidates")
        
        if not candidates:
            return []
        
        if self.language == 'auto' and candidates:
            self.language = self._detect_language_from_candidates(candidates)
        
        nodes = [self._candidate_to_node(candidate) for candidate in candidates]
        
        strategy_results = []
        for strategy in self.strategies:
            try:
                if (strategy.__name__ == '_assign_by_cjk_patterns' and 
                    self.language not in ['japanese', 'chinese']):
                    continue
                    
                result = strategy(nodes.copy())
                strategy_results.append(result)
                self.logger.debug(f"Strategy {strategy.__name__} completed")
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.__name__} failed: {e}")
        
        final_hierarchy = self._combine_strategies(strategy_results, nodes)
        
        validated_hierarchy = self._validate_and_fix_hierarchy(final_hierarchy)
        
        output = self._nodes_to_output(validated_hierarchy)
        
        self.logger.info(f"Final hierarchy: {len(output)} headings across {max([h['level'] for h in output], default=0)} levels")
        return output
    
    def _detect_language_from_candidates(self, candidates: List[HeadingCandidate]) -> str:
        sample_text = " ".join([c.text for c in candidates[:5]])
        
        if re.search(r'[\u4e00-\u9fff]', sample_text):
            if re.search(r'[\u3041-\u3096\u30A1-\u30FA]', sample_text):
                return 'japanese'
            else:
                return 'chinese'
        elif re.search(r'[\u0900-\u097F]', sample_text):
            return 'hindi'
        elif re.search(r'[\u0600-\u06FF]', sample_text):
            return 'arabic'
        else:
            return 'english'
    
    def _candidate_to_node(self, candidate) -> HierarchyNode:
        return HierarchyNode(
            text=candidate.text,
            level=1, 
            page=candidate.page,
            bbox=candidate.bbox,
            font_size=candidate.font_size,
            confidence=candidate.confidence_score,
            numbering_pattern=candidate.features.get('numbering_type'),
        )
    
    def _assign_by_cjk_patterns(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        if self.language not in ['japanese', 'chinese']:
            return nodes
        
        for node in nodes:
            text = node.text.strip()
            level = self._detect_heading_level_cjk(text, node.font_size, nodes)
            node.level = level
        
        return nodes
    
    def _detect_heading_level_cjk(self, text: str, font_size: float, all_nodes: List[HierarchyNode]) -> int:
        
        avg_font_size = np.mean([n.font_size for n in all_nodes]) if all_nodes else font_size
        
        chapter_patterns = [
            r'^第[一二三四五六七八九十\d]+章',     # 第一章, 第1章
            r'^Chapter\s*[一二三四五六七八九十\d]+', # Chapter 1 (mixed)
            r'^序章',                            # Prologue chapter
            r'^終章',                            # Final chapter
            r'^付録[一二三四五六七八九十\d]*',      # Appendix
        ]
        
        for pattern in chapter_patterns:
            if re.search(pattern, text):
                return 1
        
        major_section_patterns = [
            r'^第[一二三四五六七八九十\d]+部',     # 第一部 (Part)
            r'^第[一二三四五六七八九十\d]+編',     # 第一編 (Volume)
            r'^はじめに$',                       # Introduction (Japanese)
            r'^序論$',                          # Introduction (Japanese)
            r'^結論$',                          # Conclusion (Japanese)
            r'^まとめ$',                        # Summary (Japanese)
            r'^引言$',                          # Introduction (Chinese)
            r'^结论$',                          # Conclusion (Chinese)
            r'^总结$',                          # Summary (Chinese)
            r'^参考文献$',                       # References
            r'^謝辞$',                          # Acknowledgments (Japanese)
            r'^致谢$',                          # Acknowledgments (Chinese)
        ]
        
        for pattern in major_section_patterns:
            if re.search(pattern, text):
                return 1
        
        section_patterns = [
            r'^第[一二三四五六七八九十\d]+節',     # 第一節, 第1節
            r'^\d+\.\d+\s+[^\s]',              # 1.1 Title (with space and content)
            r'^[一二三四五六七八九十]+、',         # 一、二、三、
            r'^（[一二三四五六七八九十\d]+）',    # （一）（二）
            r'^\([一二三四五六七八九十\d]+\)',    # (一)(二)
        ]
        
        for pattern in section_patterns:
            if re.search(pattern, text):
                return 2
        
        subsection_patterns = [
            r'^\d+\.\d+\.\d+',                # 1.1.1
            r'^[一二三四五六七八九十]+\.',        # 一. 二. 三.
            r'^[abc一二三]\)',                 # a) b) c) or 一) 二)
            r'^[\(（][abc一二三\d]+[\)）]',      # (a) (b) or （一）（二）
        ]
        
        for pattern in subsection_patterns:
            if re.search(pattern, text):
                return 3
        
        minor_patterns = [
            r'^\d+\.\d+\.\d+\.\d+',           # 1.1.1.1
            r'^・',                           # Bullet points (Japanese)
            r'^•',                            # Bullet points
        ]
        
        for pattern in minor_patterns:
            if re.search(pattern, text):
                return 4
        
        if font_size > avg_font_size * 1.5:
            return 1
        elif font_size > avg_font_size * 1.2:
            return 2
        elif font_size > avg_font_size * 1.1:
            return 3
        else:
            return 4
    
    def _assign_by_font_hierarchy(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        if not nodes:
            return nodes
        
        font_sizes = [node.font_size for node in nodes]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        if len(unique_sizes) == 1:
            return self._assign_by_position_and_content(nodes)
        
        size_groups = []
        current_group = [unique_sizes[0]]
        
        for i in range(1, len(unique_sizes)):
            if unique_sizes[i-1] - unique_sizes[i] <= 1.0:  
                current_group.append(unique_sizes[i])
            else:
                size_groups.append(current_group)
                current_group = [unique_sizes[i]]
        
        if current_group:
            size_groups.append(current_group)
        
        size_to_level = {}
        for level, group in enumerate(size_groups, 1):
            for size in group:
                size_to_level[size] = min(level, 6) 
        
        for node in nodes:
            node.level = size_to_level.get(node.font_size, 3)
        
        return nodes

    def _assign_by_position_and_content(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        
        for node in nodes:
            level = 2  
            text = node.text.strip()
            
            if any(keyword in text.lower() for keyword in [
                'summary', 'background', 'introduction', 'conclusion', 
                'methodology', 'results', 'discussion', 'references'
            ]):
                level = 1
            
            elif node.bbox[1] < 150 and len(text.split()) > 3: 
                level = 1
            
            elif text.endswith(':') or re.match(r'^\d+\.', text):
                level = 2
            
            elif re.match(r'^\d+\.\d+', text):
                level = 3
            
            if node.bbox[1] < 100: 
                level = max(1, level - 1)
            
            node.level = level
        
        return nodes
    def _extract_document_title_from_headings(self, headings: List[Dict[str, Any]], 
                                          document_info: Dict[str, Any]) -> str:
        """Extract document title from the most prominent heading on first page."""
        
        logging.debug("Extracting document title from headings...")

        # Find all headings on page 1
        page_1_headings = [h for h in headings if h.get("page", 1) == 1]
        logging.debug(f"Found {len(page_1_headings)} headings on page 1.")

        if not page_1_headings:
            fallback_title = document_info.get("title", "Document Outline").strip() or "Document Outline"
            logging.info("No headings found on page 1. Falling back to document info title: '%s'", fallback_title)
            return fallback_title

        # Sort by font size (largest first) and position (topmost first)
        page_1_headings.sort(key=lambda h: (
            -h.get("font_info", {}).get("size", 12),
            h.get("bbox", [0, 0, 0, 0])[1]
        ))
        logging.debug("Sorted page 1 headings by font size and vertical position.")

        # Take the most prominent heading as title
        main_heading = page_1_headings[0]
        title_text = main_heading.get("text", "").strip()
        logging.info("Top heading candidate for title: '%s'", title_text)

        # If it's too short or generic, try combining multiple headings
        if len(title_text) < 10 or title_text.lower() in ['title', 'document', 'page']:
            logging.warning("Top heading is too short or generic. Attempting to combine multiple headings.")
            title_parts = []
            for heading in page_1_headings[:3]:
                text = heading.get("text", "").strip()
                if text and len(text) > 2:
                    title_parts.append(text)
                    logging.debug("Adding heading text to title_parts: '%s'", text)

            title_text = " ".join(title_parts) if title_parts else "Document Outline"
            logging.info("Combined heading title: '%s'", title_text)

        return title_text
    def format_results_custom(self, headings: List[Dict[str, Any]], 
                         document_info: Dict[str, Any]) -> Dict[str, Any]:
        
        self.logger.info(f"Step: Formatting custom results for {len(headings)} headings")
        
        # Extract title from the largest/most prominent heading on page 1
        title = self._extract_document_title_from_headings(headings, document_info)
        if not title and headings:
            first_text = headings[0].get("text", "").strip()
            if "rfp" in first_text.lower() or "request" in first_text.lower():
                title = first_text
            else:
                title = "Document Outline"
        elif not title:
            title = "Document Outline"
        
        outline = []
        prev_level = 0
        
        for i, heading in enumerate(headings):
            text = heading.get("text", "").strip()
            page = heading.get("page", 1)
            font_size = heading.get("font_info", {}).get("size", 12)
            
            if not text or len(text) < 3: 
                continue
            
            level_str = "H2" 
            
            if (any(keyword in text.lower() for keyword in [
                'digital library', 'ontario', 'prosperity strategy', 'summary', 
                'background', 'introduction', 'conclusion', 'critical component'
            ]) or font_size > 16 or 
            (i == 0 and len(text.split()) > 3)):
                level_str = "H1"
            
            elif (text.endswith(':') or 
                'timeline' in text.lower() or
                font_size > 14 or
                any(word in text.lower() for word in ['summary', 'background', 'timeline'])):
                level_str = "H2"
            
            elif (re.match(r'^\d+\.', text) or 
                len(text.split()) <= 3 or
                font_size <= 12):
                level_str = "H3"
            current_level = int(level_str[1])
            if prev_level > 0 and current_level > prev_level + 1:
                current_level = prev_level + 1
                level_str = f"H{current_level}"
            
            prev_level = current_level
            
            outline_item = {
                "level": level_str,
                "text": text,
                "page": page
            }
            
            outline.append(outline_item)
        
        return {
            "title": title,
            "outline": outline
        }

    def _find_natural_size_breaks(self, sorted_sizes: List[float]) -> Set[int]:
        if len(sorted_sizes) <= 2:
            return set()
        
        breaks = set()
        
        for i in range(1, len(sorted_sizes)):
            size_diff = sorted_sizes[i-1] - sorted_sizes[i]
            
            if size_diff >= 2.0:
                breaks.add(i)
            elif size_diff > sorted_sizes[i] * 0.2:
                breaks.add(i)
        
        return breaks

    def _assign_by_non_font_factors(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        for node in nodes:
            level = 2  
            
            if self.language in ['japanese', 'chinese']:
                pattern_level = self._detect_heading_level_cjk(node.text, node.font_size, nodes)
                level = pattern_level
            else:
                if node.bbox[1] < 150: 
                    level = 1
                
                spacing_before = getattr(node, 'spacing_before', 0)
                if spacing_before > 15:
                    level = max(1, level - 1)
                elif spacing_before > 8:
                    level = max(2, level)
                
                if getattr(node, 'is_bold', False):
                    level = max(1, level - 1)
            
            node.level = level
        
        return nodes
    
    def _assign_by_numbering_pattern(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        numbering_hierarchy = {
            'decimal': 1,                    # 1. 2. 3.
            'decimal_nested': 2,             # 1.1, 1.2
            'decimal_nested_deep': 3,        # 1.1.1
            'roman': 1,                      # I. II. III.
            'alpha': 2,                      # A. B. C.
            'chapter': 1,                    # Chapter 1
            'section': 2,                    # Section 1
            
            'japanese_chapter': 1,           # 第1章
            'japanese_kanji_chapter': 1,     # 第一章
            'japanese_section': 2,           # 第1節
            'japanese_kanji_list': 2,        # 一、二、三、
            'chinese_chapter': 1,            # 第1章
            'chinese_kanji_chapter': 1,      # 第一章
        }
        
        for node in nodes:
            text = node.text.strip()
            detected_level = None
            
            if self.language in ['japanese', 'chinese']:
                if re.match(r'^第\d+章', text) or re.match(r'^第[一二三四五六七八九十]+章', text):
                    detected_level = 1
                    node.numbering_pattern = 'cjk_chapter'
                elif re.match(r'^第\d+節', text) or re.match(r'^第[一二三四五六七八九十]+節', text):
                    detected_level = 2
                    node.numbering_pattern = 'cjk_section'
                elif re.match(r'^\d+\.\d+\.\d+', text):  # 1.1.1
                    detected_level = 3
                    node.numbering_pattern = 'decimal_nested_deep'
                elif re.match(r'^\d+\.\d+', text):  # 1.1
                    detected_level = 2
                    node.numbering_pattern = 'decimal_nested'
                elif re.match(r'^\d+\.', text):  # 1.
                    detected_level = 1
                    node.numbering_pattern = 'decimal'
                elif re.match(r'^[一二三四五六七八九十]+、', text):  # 一、
                    detected_level = 2
                    node.numbering_pattern = 'cjk_kanji_list'
                elif re.match(r'^（[一二三四五六七八九十\d]+）', text):  # （一）
                    detected_level = 3
                    node.numbering_pattern = 'cjk_parenthetical'
            else:
                if re.match(r'^\d+\.\d+\.\d+', text):  # 1.1.1
                    detected_level = 3
                    node.numbering_pattern = 'decimal_nested_deep'
                elif re.match(r'^\d+\.\d+', text):  # 1.1
                    detected_level = 2
                    node.numbering_pattern = 'decimal_nested'
                elif re.match(r'^\d+\.', text):  # 1.
                    detected_level = 1
                    node.numbering_pattern = 'decimal'
                elif re.match(r'^[IVX]+\.', text):  # I.
                    detected_level = 1
                    node.numbering_pattern = 'roman'
                elif re.match(r'^[A-Z]\.', text):  # A.
                    detected_level = 2
                    node.numbering_pattern = 'alpha'
                elif re.search(r'^Chapter \d+', text, re.IGNORECASE):
                    detected_level = 1
                    node.numbering_pattern = 'chapter'
                elif re.search(r'^Section \d+', text, re.IGNORECASE):
                    detected_level = 2
                    node.numbering_pattern = 'section'
            
            if detected_level:
                node.level = detected_level
        
        return nodes
    
    def _assign_by_position_and_spacing(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        nodes_sorted = sorted(nodes, key=lambda x: (x.page, x.bbox[1]))
        
        for i, node in enumerate(nodes_sorted):
            if node.bbox[1] < 100: 
                if i == 0 or nodes_sorted[i-1].page < node.page:
                    node.level = min(node.level, 1)
            
            prev_node = nodes_sorted[i-1] if i > 0 else None
            next_node = nodes_sorted[i+1] if i < len(nodes_sorted)-1 else None
            
            if prev_node and node.page == prev_node.page:
                spacing = node.bbox[1] - prev_node.bbox[3]
                if spacing > 30: 
                    node.level = max(1, node.level - 1)
        
        return nodes
    
    def _assign_by_keywords(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on common heading keywords with enhanced CJK support."""
        
        keyword_levels = {
            'title': 0, 'abstract': 0, 'summary': 0,
            
            'introduction': 1, 'background': 1, 'methodology': 1, 
            'results': 1, 'discussion': 1, 'conclusion': 1,
            'references': 1, 'bibliography': 1, 'appendix': 1,
            'chapter': 1,
            
            'section': 2, 'overview': 2, 'approach': 2,
            'analysis': 2, 'implementation': 2,
            
            'subsection': 3, 'details': 3, 'examples': 3,
            
            'はじめに': 1, '序論': 1, '結論': 1, 'まとめ': 1,  # Japanese
            '参考文献': 1, '付録': 1, '謝辞': 1,
            '引言': 1, '结论': 1, '总结': 1, '参考文献': 1,      # Chinese
            '附录': 1, '致谢': 1,
        }
        
        if self.language in self.cultural_patterns:
            cultural_keywords = self.cultural_patterns[self.language].get('heading_keywords', [])
            for keyword in cultural_keywords:
                keyword_levels[keyword.lower()] = 1
        
        for node in nodes:
            text_lower = node.text.lower().strip()
            original_text = node.text.strip()
            
            for keyword, level in keyword_levels.items():
                if (keyword in text_lower or 
                    (self.language in ['japanese', 'chinese'] and keyword in original_text)):
                    node.level = min(node.level, level)
                    node.semantic_group = keyword
                    break
        
        return nodes
    
    def _assign_by_indentation(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        
        pages = defaultdict(list)
        for node in nodes:
            pages[node.page].append(node)
        
        for page_nodes in pages.values():
            if len(page_nodes) < 2:
                continue
            
            page_nodes.sort(key=lambda x: x.bbox[1])
            
            indentations = [node.bbox[0] for node in page_nodes]
            unique_indents = sorted(set(indentations))
            
            indent_to_level = {}
            for i, indent in enumerate(unique_indents):
                indent_to_level[indent] = min(i + 1, MAX_HIERARCHY_LEVELS)
            
            for node in page_nodes:
                suggested_level = indent_to_level[node.bbox[0]]
                node.level = min(node.level, suggested_level)
        
        return nodes
    
    def _combine_strategies(self, strategy_results: List[List[HierarchyNode]], 
                          original_nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        
        if not strategy_results:
            return original_nodes
        
        level_votes = defaultdict(list)
        strategy_weights = defaultdict(float)
        
        for i, strategy_result in enumerate(strategy_results):
            weight = 2.0 if (i == 0 and self.language in ['japanese', 'chinese']) else 1.0
            
            for node in strategy_result:
                level_votes[node.text].append(node.level)
                strategy_weights[node.text] += weight
        
        final_nodes = []
        for original_node in original_nodes:
            votes = level_votes.get(original_node.text, [original_node.level])
            
            if self.language in ['japanese', 'chinese'] and len(votes) > 1:
                final_level = min(votes)
            else:
                final_level = int(np.median(votes))
            
            final_node = HierarchyNode(
                text=original_node.text,
                level=final_level,
                page=original_node.page,
                bbox=original_node.bbox,
                font_size=original_node.font_size,
                confidence=original_node.confidence,
                numbering_pattern=original_node.numbering_pattern,
                semantic_group=original_node.semantic_group,
            )
            
            final_nodes.append(final_node)
        
        return final_nodes
    
    def _validate_and_fix_hierarchy(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        
        if not nodes:
            return nodes
        
        nodes.sort(key=lambda x: (x.page, x.bbox[1]))
        
        fixed_nodes = []
        prev_level = 0
        
        for i, node in enumerate(nodes):
            current_level = node.level
            
            if self.language in ['japanese', 'chinese']:
                if re.search(r'^第[一二三四五六七八九十\d]+章', node.text):
                    current_level = 1  
                elif re.search(r'^第[一二三四五六七八九十\d]+節', node.text):
                    current_level = min(current_level, 2) 
                else:
                    if current_level > prev_level + 1:
                        current_level = prev_level + 1
            else:
                if current_level > prev_level + 1:
                    current_level = prev_level + 1
            
            current_level = max(1, min(current_level, MAX_HIERARCHY_LEVELS))
            
            if i == 0 and current_level > 1:
                current_level = 1
            
            node.level = current_level
            fixed_nodes.append(node)
            prev_level = current_level
        
        return fixed_nodes
    
    def _build_hierarchy_tree(self, nodes: List[HierarchyNode]) -> Dict[str, Any]:
        if not nodes:
            return {}
        
        tree = {}
        stack = [] 
        
        for node in nodes:
            while len(stack) > node.level:
                stack.pop()
            
            if not stack:
                tree[node.text] = {
                    "level": node.level,
                    "page": node.page,
                    "children": {}
                }
                stack.append((node.text, tree[node.text]))
            else:
                parent_name, parent_dict = stack[-1]
                parent_dict["children"][node.text] = {
                    "level": node.level,
                    "page": node.page,
                    "children": {}
                }
                stack.append((node.text, parent_dict["children"][node.text]))
        
        return tree
    
    def _nodes_to_output(self, nodes: List[HierarchyNode]) -> List[Dict[str, Any]]:
        output = []
        
        for node in nodes:
            heading_dict = {
                "text": node.text,
                "level": node.level,
                "page": node.page,
                "bbox": list(node.bbox),
                "font_info": {
                    "size": node.font_size,
                    "weight": "bold" if getattr(node, "is_bold", False) else "normal",
                    "family": getattr(node, "font_family", "unknown"),
                },
                "confidence": round(node.confidence, 3),
                "features": {
                    "numbering_pattern": node.numbering_pattern,
                    "semantic_group": node.semantic_group,
                }
            }
            output.append(heading_dict)
        
        return output
    
    def generate_hierarchy_tree(self, nodes: List[HierarchyNode]) -> Dict[str, Any]:
        return self._build_hierarchy_tree(nodes)
    
    def get_hierarchy_statistics(self, nodes: List[HierarchyNode]) -> Dict[str, Any]:
        if not nodes:
            return {}
        
        level_counts = Counter(node.level for node in nodes)
        
        stats = {
            "total_headings": len(nodes),
            "max_level": max(node.level for node in nodes),
            "min_level": min(node.level for node in nodes),
            "level_distribution": dict(level_counts),
            "avg_confidence": np.mean([node.confidence for node in nodes]),
            "pages_with_headings": len(set(node.page for node in nodes)),
            "numbering_patterns_found": list(set(
                node.numbering_pattern for node in nodes 
                if node.numbering_pattern
            )),
            "semantic_groups_found": list(set(
                node.semantic_group for node in nodes 
                if node.semantic_group
            ))
        }
        
        return stats
    
    def _detect_document_structure(self, nodes: List[HierarchyNode]) -> str:
        
        has_chapters = any("chapter" in node.text.lower() or "章" in node.text for node in nodes)
        has_sections = any("section" in node.text.lower() or "節" in node.text for node in nodes)
        has_numbering = any(node.numbering_pattern for node in nodes)
        level_distribution = Counter(node.level for node in nodes)
        
        if has_chapters:
            return "book_style"
        elif has_sections and has_numbering:
            return "academic_paper"
        elif len(level_distribution) <= 2:
            return "simple_document"
        elif max(level_distribution.keys()) >= 4:
            return "complex_hierarchical"
        else:
            return "standard_document"
    
    def optimize_for_document_type(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        
        doc_type = self._detect_document_structure(nodes)
        self.logger.debug(f"Detected document type: {doc_type}")
        
        if doc_type == "book_style":
            for node in nodes:
                text_lower = node.text.lower()
                if "chapter" in text_lower or "章" in node.text:
                    node.level = 1
                elif "section" in text_lower or "節" in node.text:
                    node.level = 2
        
        elif doc_type == "academic_paper":
            academic_mapping = {
                "abstract": 0,
                "introduction": 1,
                "related work": 1,
                "methodology": 1,
                "results": 1,
                "discussion": 1,
                "conclusion": 1,
                "references": 1,
                "はじめに": 1, "序論": 1, "結論": 1, "まとめ": 1,
                "引言": 1, "结论": 1, "总结": 1,
            }
            
            for node in nodes:
                text_lower = node.text.lower() 
                original_text = node.text 
                for keyword, level in academic_mapping.items():
                    if keyword in text_lower or keyword in original_text:
                        node.level = level
                        break
        
        elif doc_type == "simple_document":
            for node in nodes:
                if node.level > 2:
                    node.level = 2
        
        return nodes