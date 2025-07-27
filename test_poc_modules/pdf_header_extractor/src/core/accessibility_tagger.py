import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class AccessibilityTagger:
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        self.structure_types = {
            "title": "H",
            "H1": "H1", 
            "H2": "H2",
            "H3": "H3",
            "H4": "H4",
            "H5": "H5",
            "H6": "H6",
            "document": "Document",
            "section": "Sect",
            "paragraph": "P",
            "list": "L",
            "list_item": "LI"
        }
    
    def generate_pdf_ua_structure(self, headings: List[Dict]) -> Dict[str, Any]:
        self.logger.info(f"Generating PDF/UA structure for {len(headings)} headings")
        
        structure_tree = {
            "document_structure": {
                "type": "Document",
                "lang": "en-US",
                "title": self._extract_document_title(headings),
            },
            "role_map": self._generate_role_map(),
            "structure_elements": [],  
            "tagged_elements": [],
            "reading_order": [],
            "accessibility_metadata": self._generate_accessibility_metadata(headings)
        }
        
        current_section = None
        element_id = 1
        
        for heading in headings:
            element = self._create_structure_element(heading, element_id)
            structure_tree["structure_elements"].append(element) 
            structure_tree["reading_order"].append(element["id"])
            
            tagged_element = {
                "id": element["id"],
                "tag": element["structure_type"],
                "content": heading.get("text", ""),
                "page": heading.get("page", 1),
                "bbox": heading.get("bbox", [0, 0, 100, 20]),
                "attributes": {
                    "role": element["structure_type"],
                    "level": self._get_heading_level_number(heading.get("level", "H1")),
                    "lang": "en-US"
                }
            }
            structure_tree["tagged_elements"].append(tagged_element)
            
            element_id += 1
        
        structure_tree["hierarchical_structure"] = self._build_hierarchical_structure(headings)
        
        self.logger.info("PDF/UA structure generation completed")
        return structure_tree
    
    def create_structure_xml(self, headings: List[Dict]) -> str:
        self.logger.info("Creating accessibility structure XML")
        
        root = ET.Element("StructureDocument")
        root.set("xmlns", "http://www.w3.org/1999/xhtml")
        root.set("version", "1.0")
        root.set("lang", "en-US")
        
        metadata = ET.SubElement(root, "Metadata")
        self._add_metadata_elements(metadata, headings)
        
        structure_tree = ET.SubElement(root, "StructureTree")
        structure_tree.set("type", "Document")
        
        current_level = 0
        element_stack = [structure_tree]
        
        for i, heading in enumerate(headings):
            level = self._get_heading_level_number(heading.get("level", "H1"))
            text = heading.get("text", "").strip()
            page = heading.get("page", 1)
            
            while len(element_stack) > level:
                element_stack.pop()
            
            heading_elem = ET.SubElement(element_stack[-1], "StructureElement")
            heading_elem.set("type", f"H{level}")
            heading_elem.set("id", f"heading_{i+1}")
            heading_elem.set("page", str(page))
            
            content_elem = ET.SubElement(heading_elem, "Content")
            content_elem.text = text
            
            attrs_elem = ET.SubElement(heading_elem, "Attributes")
            
            role_attr = ET.SubElement(attrs_elem, "Attribute")
            role_attr.set("name", "role")
            role_attr.text = f"heading"
            
            level_attr = ET.SubElement(attrs_elem, "Attribute")
            level_attr.set("name", "aria-level")
            level_attr.text = str(level)
            
            if "bbox" in heading:
                bbox = heading["bbox"]
                bbox_elem = ET.SubElement(heading_elem, "BoundingBox")
                bbox_elem.set("x0", str(bbox[0]))
                bbox_elem.set("y0", str(bbox[1]))
                bbox_elem.set("x1", str(bbox[2]))
                bbox_elem.set("y1", str(bbox[3]))
            
            if level < 6: 
                section_elem = ET.SubElement(heading_elem, "Section")
                section_elem.set("type", "Sect")
                section_elem.set("id", f"section_{i+1}")
                element_stack.append(section_elem)
        
        nav_elem = ET.SubElement(root, "NavigationStructure")
        self._add_navigation_structure(nav_elem, headings)
        
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        
        try:
            import xml.dom.minidom
            dom = xml.dom.minidom.parseString(xml_str)
            formatted_xml = dom.toprettyxml(indent="  ")
            lines = [line for line in formatted_xml.split('\n') if line.strip()]
            return '\n'.join(lines)
        except Exception:
            return xml_str
    
    def generate_accessibility_metadata(self, headings: List[Dict]) -> Dict[str, Any]:
        
        total_headings = len(headings)
        level_distribution = {}
        
        for heading in headings:
            level = heading.get("level", "H1")
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        accessibility_issues = self._check_accessibility_issues(headings)
        
        return {
            "total_headings": total_headings,
            "level_distribution": level_distribution,
            "has_proper_hierarchy": self._check_proper_hierarchy(headings),
            "missing_h1": "H1" not in level_distribution,
            "accessibility_score": self._calculate_accessibility_score(headings),
            "issues": accessibility_issues,
            "recommendations": self._generate_recommendations(accessibility_issues),
            "compliance": {
                "wcag_2.1_aa": self._check_wcag_compliance(headings),
                "pdf_ua": self._check_pdf_ua_compliance(headings),
                "section_508": self._check_section_508_compliance(headings)
            }
        }
    
    def create_aria_labels(self, headings: List[Dict]) -> Dict[str, str]:
        
        aria_labels = {}
        
        for i, heading in enumerate(headings):
            text = heading.get("text", "").strip()
            level = self._get_heading_level_number(heading.get("level", "H1"))
            page = heading.get("page", 1)
            
            aria_label = f"Heading level {level}: {text}"
            if page > 1:
                aria_label += f" on page {page}"
            
            aria_labels[f"heading_{i+1}"] = aria_label
        
        return aria_labels
    
    def _extract_document_title(self, headings: List[Dict]) -> str:
        if not headings:
            return "Untitled Document"
        
        for heading in headings:
            level = heading.get("level", "H1")
            if level == "title" or level == "H1":
                return heading.get("text", "").strip()
        
        return headings[0].get("text", "Untitled Document").strip()
    
    def _generate_role_map(self) -> Dict[str, str]:
        return {
            "Document": "document",
            "H": "heading",
            "H1": "heading",
            "H2": "heading", 
            "H3": "heading",
            "H4": "heading",
            "H5": "heading",
            "H6": "heading",
            "Sect": "section",
            "P": "paragraph",
            "L": "list",
            "LI": "listitem"
        }
    
    def _generate_accessibility_metadata(self, headings: List[Dict]) -> Dict[str, Any]:
        return {
            "creation_date": datetime.now().isoformat(),
            "creator": "PDF-Heading-Extractor",
            "accessibility_features": [
                "structuredNavigation",
                "readingOrder", 
                "alternativeText",
                "tableHeaders"
            ],
            "language": "en-US",
            "reading_order_specified": True,
            "structure_tree_present": True,
            "heading_structure_logical": self._check_proper_hierarchy(headings)
        }
    
    def _create_structure_element(self, heading: Dict, element_id: int) -> Dict[str, Any]:
        level = heading.get("level", "H1")
        
        return {
            "id": f"struct_elem_{element_id}",
            "structure_type": self.structure_types.get(level, "H1"),
            "content": heading.get("text", "").strip(),
            "page": heading.get("page", 1),
            "bbox": heading.get("bbox", [0, 0, 100, 20]),
            "attributes": {
                "role": "heading",
                "level": self._get_heading_level_number(level),
                "lang": "en-US"
            },
            "parent_id": None,  # To be filled in hierarchical structure
            "children": []
        }
    
    def _get_heading_level_number(self, level: str) -> int:
        level_map = {
            "title": 1,
            "H1": 1,
            "H2": 2, 
            "H3": 3,
            "H4": 4,
            "H5": 5,
            "H6": 6
        }
        return level_map.get(level, 1)
    
    def _build_hierarchical_structure(self, headings: List[Dict]) -> Dict[str, Any]:
        if not headings:
            return {}
        
        root = {
            "type": "document",
            "children": [],
            "level": 0
        }
        
        stack = [root]
        
        for heading in headings:
            level = self._get_heading_level_number(heading.get("level", "H1"))
            
            while len(stack) > level:
                stack.pop()
            
            node = {
                "type": "heading",
                "level": level,
                "text": heading.get("text", "").strip(),
                "page": heading.get("page", 1),
                "children": []
            }
            
            stack[-1]["children"].append(node)
            stack.append(node)
        
        return root
    
    def _add_metadata_elements(self, metadata_elem: ET.Element, headings: List[Dict]) -> None:
        title_elem = ET.SubElement(metadata_elem, "Title")
        title_elem.text = self._extract_document_title(headings)
        
        date_elem = ET.SubElement(metadata_elem, "CreationDate")
        date_elem.text = datetime.now().isoformat()
        
        lang_elem = ET.SubElement(metadata_elem, "Language")
        lang_elem.text = "en-US"
        
        features_elem = ET.SubElement(metadata_elem, "AccessibilityFeatures")
        features = ["structuredNavigation", "readingOrder", "headingStructure"]
        for feature in features:
            feature_elem = ET.SubElement(features_elem, "Feature")
            feature_elem.text = feature
    
    def _add_navigation_structure(self, nav_elem: ET.Element, headings: List[Dict]) -> None:
        
        toc_elem = ET.SubElement(nav_elem, "TableOfContents")
        
        for i, heading in enumerate(headings):
            toc_item = ET.SubElement(toc_elem, "TOCItem")
            toc_item.set("id", f"toc_item_{i+1}")
            toc_item.set("level", str(self._get_heading_level_number(heading.get("level", "H1"))))
            toc_item.set("page", str(heading.get("page", 1)))
            toc_item.text = heading.get("text", "").strip()
    
    def _check_accessibility_issues(self, headings: List[Dict]) -> List[str]:
        issues = []
        
        if not headings:
            issues.append("No headings found in document")
            return issues
        
        has_h1 = any(h.get("level") == "H1" for h in headings)
        if not has_h1:
            issues.append("Missing H1 heading - document should start with H1")
        
        levels_used = set()
        for heading in headings:
            level_num = self._get_heading_level_number(heading.get("level", "H1"))
            levels_used.add(level_num)
        
        sorted_levels = sorted(levels_used)
        for i in range(1, len(sorted_levels)):
            if sorted_levels[i] - sorted_levels[i-1] > 1:
                issues.append(f"Heading level skipped: jumped from H{sorted_levels[i-1]} to H{sorted_levels[i]}")
        
        empty_headings = [h for h in headings if not h.get("text", "").strip()]
        if empty_headings:
            issues.append(f"Found {len(empty_headings)} empty headings")
        
        return issues
    
    def _check_proper_hierarchy(self, headings: List[Dict]) -> bool:
        if not headings:
            return True
        
        prev_level = 0
        for heading in headings:
            level = self._get_heading_level_number(heading.get("level", "H1"))
            
            if level - prev_level > 1:
                return False
            
            prev_level = level
        
        return True
    
    def _calculate_accessibility_score(self, headings: List[Dict]) -> float:
        if not headings:
            return 0.0
        
        score = 100.0
        issues = self._check_accessibility_issues(headings)
        
        score -= len(issues) * 10
        
        if any(h.get("level") == "H1" for h in headings):
            score += 5
        
        if self._check_proper_hierarchy(headings):
            score += 10
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        recommendations = []
        
        for issue in issues:
            if "Missing H1" in issue:
                recommendations.append("Add an H1 heading at the beginning of the document")
            elif "skipped" in issue:
                recommendations.append("Use sequential heading levels (H1, H2, H3) without skipping")
            elif "empty" in issue:
                recommendations.append("Provide meaningful text for all headings")
        
        if not recommendations:
            recommendations.append("Document structure follows accessibility best practices")
        
        return recommendations
    
    def _check_wcag_compliance(self, headings: List[Dict]) -> bool:
        issues = self._check_accessibility_issues(headings)
        return len(issues) == 0 and self._check_proper_hierarchy(headings)
    
    def _check_pdf_ua_compliance(self, headings: List[Dict]) -> bool:
        return self._check_wcag_compliance(headings) and len(headings) > 0
    
    def _check_section_508_compliance(self, headings: List[Dict]) -> bool:
        return self._check_wcag_compliance(headings)