import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from candidate_generator import HeadingCandidate
import numpy as np
from config.settings import MAX_HIERARCHY_LEVELS, TITLE_POSITION_THRESHOLD
from config.cultural_patterns import CULTURAL_PATTERNS


@dataclass
class HierarchyNode:
    """Represents a node in the heading hierarchy."""
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
    """Assigns hierarchy levels to heading candidates using multiple strategies."""
    
    def __init__(self, language: str = 'auto', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.cultural_patterns = CULTURAL_PATTERNS
        
        # Hierarchy detection strategies
        self.strategies = [
            self._assign_by_font_hierarchy,
            self._assign_by_numbering_pattern,
            self._assign_by_position_and_spacing,
            self._assign_by_keywords,
            self._assign_by_indentation,
        ]
    
    def assign_hierarchy(self, candidates: List[HeadingCandidate]) -> List[Dict[str, Any]]:
        """Assign hierarchy levels to heading candidates."""
        self.logger.info(f"Assigning hierarchy to {len(candidates)} candidates")
        
        if not candidates:
            return []
        
        # Convert candidates to hierarchy nodes
        nodes = [self._candidate_to_node(candidate) for candidate in candidates]
        
        # Apply multiple strategies and combine results
        strategy_results = []
        for strategy in self.strategies:
            try:
                result = strategy(nodes.copy())
                strategy_results.append(result)
                self.logger.debug(f"Strategy {strategy.__name__} completed")
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.__name__} failed: {e}")
        
        # Combine strategies using ensemble approach
        final_hierarchy = self._combine_strategies(strategy_results, nodes)
        
        # Post-process and validate hierarchy
        validated_hierarchy = self._validate_and_fix_hierarchy(final_hierarchy)
        
        # Convert back to output format
        output = self._nodes_to_output(validated_hierarchy)
        
        self.logger.info(f"Final hierarchy: {len(output)} headings across {max([h['level'] for h in output], default=0)} levels")
        return output
    
    def _candidate_to_node(self, candidate) -> HierarchyNode:
        """Convert heading candidate to hierarchy node."""
        return HierarchyNode(
            text=candidate.text,
            level=1,  # Default level, will be updated
            page=candidate.page,
            bbox=candidate.bbox,
            font_size=candidate.font_size,
            confidence=candidate.confidence_score,
            numbering_pattern=candidate.features.get('numbering_type'),
        )
    
    def _assign_by_font_hierarchy(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on font size hierarchy."""
        if not nodes:
            return nodes
        
        # Group by font size
        font_sizes = sorted(set(node.font_size for node in nodes), reverse=True)
        
        # Create level mapping based on font sizes
        level_mapping = {}
        for i, size in enumerate(font_sizes[:MAX_HIERARCHY_LEVELS]):
            level_mapping[size] = i + 1
        
        # Special handling for title (largest font at top of document)
        title_candidates = [
            node for node in nodes 
            if (node.font_size == max(font_sizes) and 
                node.bbox[1] / 792 < TITLE_POSITION_THRESHOLD)  # Assuming A4 height
        ]
        
        for node in nodes:
            if node in title_candidates and len(title_candidates) <= 2:
                node.level = 0  # Title level
            else:
                node.level = level_mapping.get(node.font_size, MAX_HIERARCHY_LEVELS)
        
        return nodes
    
    def _assign_by_numbering_pattern(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on numbering patterns."""
        numbering_hierarchy = {
            'decimal': 1,      # 1. 2. 3.
            'decimal_nested': 2,  # 1.1, 1.2
            'roman': 1,        # I. II. III.
            'alpha': 2,        # A. B. C.
            'chapter': 1,      # Chapter 1
            'section': 2,   
            'decimal_nested_deep': 3,
        }
        
        for node in nodes:
            text = node.text.strip()
            
            # Detect numbering patterns
            if re.match(r'^\d+\.\d+\.\d+', text):  # 1.1.1
                node.level = 3
                node.numbering_pattern = 'decimal_nested_deep'
            elif re.match(r'^\d+\.\d+', text):  # 1.1
                node.level = 2
                node.numbering_pattern = 'decimal_nested'
            elif re.match(r'^\d+\.', text):  # 1.
                node.level = 1
                node.numbering_pattern = 'decimal'
            elif re.match(r'^[IVX]+\.', text):  # I.
                node.level = 1
                node.numbering_pattern = 'roman'
            elif re.match(r'^[A-Z]\.', text):  # A.
                node.level = 2
                node.numbering_pattern = 'alpha'
            elif re.search(r'^Chapter \d+', text, re.IGNORECASE):
                node.level = 1
                node.numbering_pattern = 'chapter'
            elif re.search(r'^Section \d+', text, re.IGNORECASE):
                node.level = 2
                node.numbering_pattern = 'section'
            else:
                # No clear numbering, use font-based fallback
                continue
        
        return nodes
    
    def _assign_by_position_and_spacing(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on position and spacing patterns."""
        
        # Sort nodes by page and position
        nodes_sorted = sorted(nodes, key=lambda x: (x.page, x.bbox[1]))
        
        for i, node in enumerate(nodes_sorted):
            # Check if it's at the very top of a page (likely title/chapter)
            if node.bbox[1] < 100:  # Top 100 points of page
                if i == 0 or nodes_sorted[i-1].page < node.page:
                    node.level = min(node.level, 1)
            
            # Analyze spacing context
            prev_node = nodes_sorted[i-1] if i > 0 else None
            next_node = nodes_sorted[i+1] if i < len(nodes_sorted)-1 else None
            
            # Large spacing before suggests higher level
            if prev_node and node.page == prev_node.page:
                spacing = node.bbox[1] - prev_node.bbox[3]
                if spacing > 30:  # Significant spacing
                    node.level = max(1, node.level - 1)
        
        return nodes
    
    def _assign_by_keywords(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on common heading keywords."""
        
        keyword_levels = {
            # Level 0 (Title)
            'title': 0, 'abstract': 0, 'summary': 0,
            
            # Level 1 (Major sections)
            'introduction': 1, 'background': 1, 'methodology': 1, 
            'results': 1, 'discussion': 1, 'conclusion': 1,
            'references': 1, 'bibliography': 1, 'appendix': 1,
            'chapter': 1,
            
            # Level 2 (Subsections)
            'section': 2, 'overview': 2, 'approach': 2,
            'analysis': 2, 'implementation': 2,
            
            # Level 3 (Sub-subsections)
            'subsection': 3, 'details': 3, 'examples': 3,
        }
        
        # Add cultural keywords
        if self.language in self.cultural_patterns:
            cultural_keywords = self.cultural_patterns[self.language].get('heading_keywords', [])
            for keyword in cultural_keywords:
                keyword_levels[keyword.lower()] = 1
        
        for node in nodes:
            text_lower = node.text.lower().strip()
            
            # Check for exact matches first
            for keyword, level in keyword_levels.items():
                if keyword in text_lower:
                    node.level = min(node.level, level)
                    node.semantic_group = keyword
                    break
        
        return nodes
    
    def _assign_by_indentation(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on indentation patterns."""
        
        # Group nodes by page to analyze indentation within pages
        pages = defaultdict(list)
        for node in nodes:
            pages[node.page].append(node)
        
        for page_nodes in pages.values():
            if len(page_nodes) < 2:
                continue
            
            # Sort by vertical position
            page_nodes.sort(key=lambda x: x.bbox[1])
            
            # Analyze indentation levels
            indentations = [node.bbox[0] for node in page_nodes]
            unique_indents = sorted(set(indentations))
            
            # Map indentation to hierarchy levels
            indent_to_level = {}
            for i, indent in enumerate(unique_indents):
                indent_to_level[indent] = min(i + 1, MAX_HIERARCHY_LEVELS)
            
            # Assign levels based on indentation
            for node in page_nodes:
                suggested_level = indent_to_level[node.bbox[0]]
                # Take minimum with existing level (most restrictive)
                node.level = min(node.level, suggested_level)
        
        return nodes
    
    def _combine_strategies(self, strategy_results: List[List[HierarchyNode]], 
                          original_nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Combine results from multiple strategies using ensemble approach."""
        
        if not strategy_results:
            return original_nodes
        
        # Create mapping from text to level votes
        level_votes = defaultdict(list)
        
        for strategy_result in strategy_results:
            for node in strategy_result:
                level_votes[node.text].append(node.level)
        
        # Assign final levels using median voting
        final_nodes = []
        for original_node in original_nodes:
            votes = level_votes.get(original_node.text, [original_node.level])
            
            # Use median of votes, with bias towards lower levels (higher importance)
            final_level = int(np.median(votes))
            
            # Create final node
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
        """Validate and fix common hierarchy issues."""
        
        if not nodes:
            return nodes
        
        # Sort nodes by page and position
        nodes.sort(key=lambda x: (x.page, x.bbox[1]))
        
        # Fix common issues
        fixed_nodes = []
        prev_level = 0
        
        for i, node in enumerate(nodes):
            current_level = node.level
            
            # Ensure we don't skip levels (max jump of 1)
            if current_level > prev_level + 1:
                current_level = prev_level + 1
            
            # Ensure reasonable bounds
            current_level = max(0, min(current_level, MAX_HIERARCHY_LEVELS))
            
            # Special case: if first heading is not level 1, make it level 1
            if i == 0 and current_level > 1:
                current_level = 1
            
            # Update node
            node.level = current_level
            fixed_nodes.append(node)
            prev_level = current_level
        
        return fixed_nodes
    
    def _build_hierarchy_tree(self, nodes: List[HierarchyNode]) -> Dict[str, Any]:
        """Build a hierarchical tree structure."""
        if not nodes:
            return {}
        
        tree = {}
        stack = []  # Stack to track parent nodes at each level
        
        for node in nodes:
            # Adjust stack size to current level
            while len(stack) > node.level:
                stack.pop()
            
            # Add current node
            if not stack:
                # Root level
                tree[node.text] = {
                    "level": node.level,
                    "page": node.page,
                    "children": {}
                }
                stack.append((node.text, tree[node.text]))
            else:
                # Child node
                parent_name, parent_dict = stack[-1]
                parent_dict["children"][node.text] = {
                    "level": node.level,
                    "page": node.page,
                    "children": {}
                }
                stack.append((node.text, parent_dict["children"][node.text]))
        
        return tree
    
    def _nodes_to_output(self, nodes: List[HierarchyNode]) -> List[Dict[str, Any]]:
        """Convert hierarchy nodes to output format."""
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
        """Generate a hierarchical tree representation."""
        return self._build_hierarchy_tree(nodes)
    
    def get_hierarchy_statistics(self, nodes: List[HierarchyNode]) -> Dict[str, Any]:
        """Generate statistics about the detected hierarchy."""
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
        """Detect the overall document structure type."""
        
        # Analyze patterns
        has_chapters = any("chapter" in node.text.lower() for node in nodes)
        has_sections = any("section" in node.text.lower() for node in nodes)
        has_numbering = any(node.numbering_pattern for node in nodes)
        level_distribution = Counter(node.level for node in nodes)
        
        # Classify document type
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
        """Optimize hierarchy based on detected document type."""
        
        doc_type = self._detect_document_structure(nodes)
        self.logger.debug(f"Detected document type: {doc_type}")
        
        if doc_type == "book_style":
            # Ensure chapters are level 1, sections are level 2
            for node in nodes:
                if "chapter" in node.text.lower():
                    node.level = 1
                elif "section" in node.text.lower():
                    node.level = 2
        
        elif doc_type == "academic_paper":
            # Standard academic structure
            academic_mapping = {
                "abstract": 0,
                "introduction": 1,
                "related work": 1,
                "methodology": 1,
                "results": 1,
                "discussion": 1,
                "conclusion": 1,
                "references": 1,
            }
            
            for node in nodes:
                text_lower = node.text.lower()
                for keyword, level in academic_mapping.items():
                    if keyword in text_lower:
                        node.level = level
                        break
        
        elif doc_type == "simple_document":
            # Flatten hierarchy for simple documents
            for node in nodes:
                if node.level > 2:
                    node.level = 2
        
        return nodes