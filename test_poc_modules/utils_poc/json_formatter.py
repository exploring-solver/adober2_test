# poc-modules/utils_poc/json_formatter.py

import json
from datetime import datetime
from typing import Dict, List, Any

class JSONFormatter:
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
    
    def format_round1a_output(self, title: str, outline: List[Dict]) -> Dict:
        """Format Round 1A output according to challenge specification"""
        return {
            "title": title,
            "outline": [
                {
                    "level": item["level"],
                    "text": item["text"],
                    "page": item["page"]
                }
                for item in outline
            ]
        }
    
    def format_round1b_output(self, 
                            input_documents: List[str],
                            persona: str,
                            job_to_be_done: str,
                            sections: List[Dict],
                            sub_sections: List[Dict]) -> Dict:
        """Format Round 1B output according to challenge specification"""
        
        # Format extracted sections
        extracted_sections = []
        for section in sections[:10]:  # Top 10 sections
            extracted_sections.append({
                "document": section["document"],
                "page_number": section["page_number"],
                "section_title": section["section_title"],
                "importance_rank": section["importance_rank"]
            })
        
        # Format sub-section analysis
        subsection_analysis = []
        for sub in sub_sections[:15]:  # Top 15 sub-sections
            subsection_analysis.append({
                "document": sub["document"],
                "page_number": sub["page_number"],
                "refined_text": sub["refined_text"],
                "parent_section": sub.get("parent_section", "")
            })
        
        return {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": self.timestamp
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
    
    def save_json(self, data: Dict, filename: str) -> bool:
        """Save JSON data to file with proper formatting"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    def validate_round1a_format(self, data: Dict) -> tuple[bool, List[str]]:
        """Validate Round 1A JSON format"""
        errors = []
        
        # Check required top-level fields
        if "title" not in data:
            errors.append("Missing 'title' field")
        
        if "outline" not in data:
            errors.append("Missing 'outline' field")
            return False, errors
        
        if not isinstance(data["outline"], list):
            errors.append("'outline' must be a list")
            return False, errors
        
        # Validate outline items
        for i, item in enumerate(data["outline"]):
            if not isinstance(item, dict):
                errors.append(f"Outline item {i} must be a dictionary")
                continue
            
            required_fields = ["level", "text", "page"]
            for field in required_fields:
                if field not in item:
                    errors.append(f"Outline item {i} missing '{field}' field")
            
            if "level" in item and item["level"] not in ["H1", "H2", "H3"]:
                errors.append(f"Outline item {i} has invalid level: {item['level']}")
            
            if "page" in item and not isinstance(item["page"], int):
                errors.append(f"Outline item {i} page must be integer")
        
        return len(errors) == 0, errors
    
    def validate_round1b_format(self, data: Dict) -> tuple[bool, List[str]]:
        """Validate Round 1B JSON format"""
        errors = []
        
        # Check metadata
        if "metadata" not in data:
            errors.append("Missing 'metadata' field")
        else:
            metadata = data["metadata"]
            required_meta_fields = ["input_documents", "persona", "job_to_be_done", "processing_timestamp"]
            for field in required_meta_fields:
                if field not in metadata:
                    errors.append(f"Missing metadata field: {field}")
        
        # Check extracted sections
        if "extracted_sections" not in data:
            errors.append("Missing 'extracted_sections' field")
        else:
            for i, section in enumerate(data["extracted_sections"]):
                required_section_fields = ["document", "page_number", "section_title", "importance_rank"]
                for field in required_section_fields:
                    if field not in section:
                        errors.append(f"Section {i} missing field: {field}")
        
        # Check subsection analysis
        if "subsection_analysis" not in data:
            errors.append("Missing 'subsection_analysis' field")
        else:
            for i, subsection in enumerate(data["subsection_analysis"]):
                required_sub_fields = ["document", "page_number", "refined_text"]
                for field in required_sub_fields:
                    if field not in subsection:
                        errors.append(f"Subsection {i} missing field: {field}")
        
        return len(errors) == 0, errors
    
    def create_sample_outputs(self):
        """Create sample output files for reference"""
        
        # Sample Round 1A output
        sample_1a = self.format_round1a_output(
            title="Understanding AI",
            outline=[
                {"level": "H1", "text": "Introduction", "page": 1},
                {"level": "H2", "text": "What is AI?", "page": 2},
                {"level": "H3", "text": "History of AI", "page": 3}
            ]
        )
        
        # Sample Round 1B output
        sample_1b = self.format_round1b_output(
            input_documents=["doc1.pdf", "doc2.pdf"],
            persona="PhD Researcher in Computational Biology",
            job_to_be_done="Prepare comprehensive literature review",
            sections=[
                {
                    "document": "doc1.pdf",
                    "page_number": 1,
                    "section_title": "Introduction",
                    "importance_rank": 1
                }
            ],
            sub_sections=[
                {
                    "document": "doc1.pdf",
                    "page_number": 1,
                    "refined_text": "Sample refined text content here",
                    "parent_section": "Introduction"
                }
            ]
        )
        
        # Save samples
        self.save_json(sample_1a, "sample_round1a_output.json")
        self.save_json(sample_1b, "sample_round1b_output.json")
        
        return sample_1a, sample_1b

# Test the formatter
if __name__ == "__main__":
    print("JSON Formatter PoC")
    print("=" * 40)
    
    formatter = JSONFormatter()
    
    # Create and validate sample outputs
    sample_1a, sample_1b = formatter.create_sample_outputs()
    
    print("Round 1A Sample Output:")
    print("-" * 25)
    print(json.dumps(sample_1a, indent=2))
    
    # Validate Round 1A
    is_valid_1a, errors_1a = formatter.validate_round1a_format(sample_1a)
    print(f"\nRound 1A Validation: {'✓ PASS' if is_valid_1a else '✗ FAIL'}")
    if errors_1a:
        for error in errors_1a:
            print(f"  - {error}")
    
    print("\n" + "="*50)
    print("Round 1B Sample Output:")
    print("-" * 25)
    print(json.dumps(sample_1b, indent=2))
    
    # Validate Round 1B
    is_valid_1b, errors_1b = formatter.validate_round1b_format(sample_1b)
    print(f"\nRound 1B Validation: {'✓ PASS' if is_valid_1b else '✗ FAIL'}")
    if errors_1b:
        for error in errors_1b:
            print(f"  - {error}")
    
    print(f"\nSample files saved:")
    print("- sample_round1a_output.json")
    print("- sample_round1b_output.json")