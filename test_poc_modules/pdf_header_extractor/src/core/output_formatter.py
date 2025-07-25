import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import asdict
import csv
from config.settings import OUTPUT_FORMAT, INCLUDE_CONFIDENCE_SCORES, INCLUDE_DEBUG_INFO


class OutputFormatter:
    """Formats and exports PDF heading extraction results in multiple formats."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
    def format_results(self, headings: List[Dict[str, Any]], 
                      document_info: Dict[str, Any],
                      hierarchy_tree: Optional[Dict[str, Any]] = None,
                      processing_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format extraction results into standardized output."""
        
        # Build base result structure
        result = {
            "document_info": self._format_document_info(document_info),
            "headings": self._format_headings(headings),
            "metadata": self._generate_metadata(),
        }
        
        # Add optional components
        if hierarchy_tree:
            result["hierarchy_tree"] = hierarchy_tree
            
        if processing_stats:
            result["processing_statistics"] = processing_stats
            
        if self.debug and INCLUDE_DEBUG_INFO:
            result["debug_info"] = self._generate_debug_info(headings, document_info)
        
        # Validate result structure
        self._validate_output(result)
        
        return result
    
    def _format_document_info(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format document information section."""
        formatted_info = {
            "filename": document_info.get("filename", "unknown.pdf"),
            "total_pages": document_info.get("total_pages", 0),
            "processing_time": round(document_info.get("processing_time", 0), 3),
            "file_size_mb": round(document_info.get("file_size", 0) / (1024 * 1024), 2),
            "language_detected": document_info.get("language", "auto"),
            "extraction_method": document_info.get("method", "hybrid"),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add optional fields if available
        optional_fields = ["title", "author", "subject", "creator", "creation_date"]
        for field in optional_fields:
            if field in document_info:
                formatted_info[field] = document_info[field]
        
        return formatted_info
    
    def _format_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format headings list with consistent structure."""
        formatted_headings = []
        
        for i, heading in enumerate(headings):
            formatted_heading = {
                "id": i + 1,
                "text": heading.get("text", "").strip(),
                "level": max(0, min(heading.get("level", 1), 6)),  # Clamp 0-6
                "page": heading.get("page", 1),
                "bbox": self._format_bbox(heading.get("bbox", [0, 0, 0, 0])),
                "font_info": self._format_font_info(heading.get("font_info", {})),
            }
            
            # Add confidence if enabled
            if INCLUDE_CONFIDENCE_SCORES:
                formatted_heading["confidence"] = round(
                    heading.get("confidence", 0.0), 3
                )
            
            # Add optional features
            features = heading.get("features", {})
            if features:
                formatted_heading["features"] = self._format_features(features)
            
            formatted_headings.append(formatted_heading)
        
        return formatted_headings
    
    def _format_bbox(self, bbox: List[float]) -> Dict[str, float]:
        """Format bounding box coordinates."""
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        
        return {
            "x0": round(bbox[0], 2),
            "y0": round(bbox[1], 2),
            "x1": round(bbox[2], 2),
            "y1": round(bbox[3], 2),
            "width": round(bbox[2] - bbox[0], 2),
            "height": round(bbox[3] - bbox[1], 2)
        }
    
    def _format_font_info(self, font_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format font information."""
        return {
            "size": font_info.get("size", 12),
            "weight": font_info.get("weight", "normal"),
            "family": font_info.get("family", "unknown"),
            "style": font_info.get("style", "normal")
        }
    
    def _format_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Format heading features and metadata."""
        formatted_features = {}
        
        # Numbering information
        if features.get("numbering_pattern"):
            formatted_features["numbering"] = {
                "pattern": features["numbering_pattern"],
                "has_numbering": True
            }
        
        # Semantic information
        if features.get("semantic_group"):
            formatted_features["semantic"] = {
                "group": features["semantic_group"],
                "keywords_matched": features.get("keywords_matched", [])
            }
        
        # Layout information
        layout_features = {}
        if "alignment" in features:
            layout_features["alignment"] = features["alignment"]
        if "indentation" in features:
            layout_features["indentation"] = round(features["indentation"], 2)
        if "spacing_before" in features:
            layout_features["spacing_before"] = round(features["spacing_before"], 2)
        if "spacing_after" in features:
            layout_features["spacing_after"] = round(features["spacing_after"], 2)
        
        if layout_features:
            formatted_features["layout"] = layout_features
        
        return formatted_features
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate extraction metadata."""
        return {
            "extraction_version": "1.0.0",
            "format_version": "1.0",
            "generator": "PDF-Heading-Extractor-Hybrid",
            "extraction_date": datetime.now().isoformat(),
            "settings": {
                "include_confidence": INCLUDE_CONFIDENCE_SCORES,
                "include_debug": INCLUDE_DEBUG_INFO,
                "output_format": OUTPUT_FORMAT
            }
        }
    
    def _generate_debug_info(self, headings: List[Dict[str, Any]], 
                           document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate debug information for troubleshooting."""
        return {
            "total_candidates_found": len(headings),
            "level_distribution": self._calculate_level_distribution(headings),
            "confidence_distribution": self._calculate_confidence_distribution(headings),
            "page_distribution": self._calculate_page_distribution(headings),
            "font_analysis": document_info.get("font_analysis", {}),
            "processing_stages": document_info.get("processing_stages", []),
            "warnings": document_info.get("warnings", [])
        }
    
    def _calculate_level_distribution(self, headings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of heading levels."""
        distribution = {}
        for heading in headings:
            level = heading.get("level", 1)
            distribution[f"level_{level}"] = distribution.get(f"level_{level}", 0) + 1
        return distribution
    
    def _calculate_confidence_distribution(self, headings: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence score statistics."""
        if not headings:
            return {}
        
        confidences = [h.get("confidence", 0.0) for h in headings]
        return {
            "min": round(min(confidences), 3),
            "max": round(max(confidences), 3),
            "mean": round(sum(confidences) / len(confidences), 3),
            "median": round(sorted(confidences)[len(confidences)//2], 3)
        }
    
    def _calculate_page_distribution(self, headings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of headings across pages."""
        distribution = {}
        for heading in headings:
            page = heading.get("page", 1)
            distribution[f"page_{page}"] = distribution.get(f"page_{page}", 0) + 1
        return distribution
    
    def _validate_output(self, result: Dict[str, Any]) -> None:
        """Validate output structure and content."""
        required_fields = ["document_info", "headings", "metadata"]
        
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate headings structure
        for i, heading in enumerate(result["headings"]):
            required_heading_fields = ["id", "text", "level", "page", "bbox"]
            for field in required_heading_fields:
                if field not in heading:
                    raise ValueError(f"Heading {i+1} missing required field: {field}")
        
        self.logger.debug("Output validation passed")
    
    def save_json(self, result: Dict[str, Any], output_path: str, 
                  pretty: bool = True) -> None:
        """Save results as JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                json.dump(result, f, ensure_ascii=False)
        
        self.logger.info(f"Results saved to JSON: {output_path}")
    
    def save_csv(self, result: Dict[str, Any], output_path: str) -> None:
        """Save headings as CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        headings = result["headings"]
        
        fieldnames = [
            "id", "text", "level", "page", "confidence",
            "font_size", "font_weight", "bbox_x0", "bbox_y0", 
            "bbox_x1", "bbox_y1", "width", "height"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for heading in headings:
                row = {
                    "id": heading["id"],
                    "text": heading["text"],
                    "level": heading["level"],
                    "page": heading["page"],
                    "confidence": heading.get("confidence", 0.0),
                    "font_size": heading["font_info"]["size"],
                    "font_weight": heading["font_info"]["weight"],
                    "bbox_x0": heading["bbox"]["x0"],
                    "bbox_y0": heading["bbox"]["y0"],
                    "bbox_x1": heading["bbox"]["x1"],
                    "bbox_y1": heading["bbox"]["y1"],
                    "width": heading["bbox"]["width"],
                    "height": heading["bbox"]["height"]
                }
                writer.writerow(row)
        
        self.logger.info(f"Results saved to CSV: {output_path}")
    
    def save_xml(self, result: Dict[str, Any], output_path: str) -> None:
        """Save results as XML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        root = ET.Element("pdf_headings")
        
        # Document info
        doc_info = ET.SubElement(root, "document_info")
        for key, value in result["document_info"].items():
            elem = ET.SubElement(doc_info, key)
            elem.text = str(value)
        
        # Headings
        headings_elem = ET.SubElement(root, "headings")
        for heading in result["headings"]:
            heading_elem = ET.SubElement(headings_elem, "heading")
            heading_elem.set("id", str(heading["id"]))
            heading_elem.set("level", str(heading["level"]))
            heading_elem.set("page", str(heading["page"]))
            
            # Text content
            text_elem = ET.SubElement(heading_elem, "text")
            text_elem.text = heading["text"]
            
            # Bounding box
            bbox_elem = ET.SubElement(heading_elem, "bbox")
            for coord, value in heading["bbox"].items():
                coord_elem = ET.SubElement(bbox_elem, coord)
                coord_elem.text = str(value)
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        self.logger.info(f"Results saved to XML: {output_path}")
    
    def save_markdown(self, result: Dict[str, Any], output_path: str) -> None:
        """Save headings as Markdown outline."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        
        # Header
        lines.append(f"# Document Outline: {result['document_info']['filename']}")
        lines.append("")
        lines.append(f"- **Total Pages:** {result['document_info']['total_pages']}")
        lines.append(f"- **Processing Time:** {result['document_info']['processing_time']}s")
        lines.append(f"- **Total Headings:** {len(result['headings'])}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Headings outline
        for heading in result["headings"]:
            level = heading["level"]
            text = heading["text"]
            page = heading["page"]
            
            # Create markdown heading
            if level == 0:
                lines.append(f"# {text} *(Page {page})*")
            else:
                indent = "  " * (level - 1)
                lines.append(f"{indent}- **{text}** *(Page {page})*")
        
        # Write markdown
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        self.logger.info(f"Results saved to Markdown: {output_path}")
    
    def save_html_outline(self, result: Dict[str, Any], output_path: str) -> None:
        """Save headings as HTML outline with collapsible tree."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Outline: {filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .outline {{ max-width: 800px; }}
        .heading {{ margin: 5px 0; padding: 5px; border-left: 3px solid #007acc; }}
        .level-0 {{ font-size: 1.5em; font-weight: bold; color: #333; }}
        .level-1 {{ font-size: 1.3em; font-weight: bold; margin-left: 0px; }}
        .level-2 {{ font-size: 1.1em; font-weight: bold; margin-left: 20px; }}
        .level-3 {{ font-size: 1.0em; margin-left: 40px; }}
        .level-4 {{ font-size: 0.9em; margin-left: 60px; }}
        .level-5 {{ font-size: 0.9em; margin-left: 80px; }}
        .level-6 {{ font-size: 0.8em; margin-left: 100px; }}
        .page-num {{ color: #666; font-size: 0.8em; }}
        .confidence {{ color: #999; font-size: 0.7em; }}
        .stats {{ background: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>PDF Document Outline</h1>
    
    <div class="stats">
        <h3>Document Information</h3>
        <p><strong>File:</strong> {filename}</p>
        <p><strong>Pages:</strong> {total_pages}</p>
        <p><strong>Processing Time:</strong> {processing_time}s</p>
        <p><strong>Total Headings:</strong> {total_headings}</p>
        <p><strong>Language:</strong> {language}</p>
    </div>
    
    <div class="outline">
        <h3>Heading Structure</h3>
        {headings_html}
    </div>
</body>
</html>"""
        
        # Generate headings HTML
        headings_html = []
        for heading in result["headings"]:
            level = heading["level"]
            text = heading["text"]
            page = heading["page"]
            confidence = heading.get("confidence", 0.0)
            
            confidence_str = f'<span class="confidence">(conf: {confidence:.2f})</span>' if INCLUDE_CONFIDENCE_SCORES else ''
            
            heading_html = f'''
            <div class="heading level-{level}">
                {text} 
                <span class="page-num">Page {page}</span>
                {confidence_str}
            </div>'''
            
            headings_html.append(heading_html)
        
        # Fill template
        html_content = html_template.format(
            filename=result["document_info"]["filename"],
            total_pages=result["document_info"]["total_pages"],
            processing_time=result["document_info"]["processing_time"],
            total_headings=len(result["headings"]),
            language=result["document_info"]["language_detected"],
            headings_html="".join(headings_html)
        )
        
        # Write HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Results saved to HTML: {output_path}")
    
    def export_multiple_formats(self, result: Dict[str, Any], 
                               base_path: str, formats: List[str]) -> Dict[str, str]:
        """Export results in multiple formats."""
        base_path = Path(base_path)
        output_files = {}
        
        for format_type in formats:
            if format_type == "json":
                output_path = base_path.with_suffix('.json')
                self.save_json(result, output_path)
                output_files["json"] = str(output_path)
                
            elif format_type == "csv":
                output_path = base_path.with_suffix('.csv')
                self.save_csv(result, output_path)
                output_files["csv"] = str(output_path)
                
            elif format_type == "xml":
                output_path = base_path.with_suffix('.xml')
                self.save_xml(result, output_path)
                output_files["xml"] = str(output_path)
                
            elif format_type == "markdown":
                output_path = base_path.with_suffix('.md')
                self.save_markdown(result, output_path)
                output_files["markdown"] = str(output_path)
                
            elif format_type == "html":
                output_path = base_path.with_suffix('.html')
                self.save_html_outline(result, output_path)
                output_files["html"] = str(output_path)
        
        return output_files