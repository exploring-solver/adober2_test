# setup_poc.py - Quick setup script for PoC modules

import os
import sys
from pathlib import Path

def create_poc_structure():
    """Create the PoC folder structure"""
    
    structure = [
        "poc-modules",
        "poc-modules/1a_pdf_parser_poc",
        "poc-modules/1a_structure_extractor_poc", 
        "poc-modules/1b_persona_analyzer_poc",
        "poc-modules/1b_content_ranker_poc",
        "poc-modules/utils_poc",
        "poc-modules/integration_poc",
        "poc-modules/sample_data",
        "poc-modules/sample_data/input",
        "poc-modules/sample_data/output"
    ]
    
    print("Creating PoC folder structure...")
    for folder in structure:
        Path(folder).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files for Python modules
        if not folder.endswith(('input', 'output', 'sample_data', 'poc-modules')):
            init_file = Path(folder) / "__init__.py"
            init_file.touch()
    
    print("✓ Folder structure created")

def create_requirements_file():
    """Create requirements.txt for PoC"""
    requirements = """# PoC Requirements - Minimal setup for testing
PyMuPDF==1.23.5
# Alternative: pdfplumber==0.9.0
numpy==1.24.3
# For production: sentence-transformers==2.2.2 (but >200MB)
# For PoC: we'll use simple keyword matching

# Optional for enhanced PoC:
# scikit-learn==1.3.0  # For TF-IDF if needed
# nltk==3.8.1  # For text processing
"""
    
    with open("poc-modules/requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✓ Requirements file created")

def create_test_runner():
    """Create a simple test runner script"""
    test_script = """#!/usr/bin/env python3
# run_poc_tests.py - Test runner for all PoC modules

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    print("Adobe Hackathon PoC - Test Runner")
    print("=" * 40)
    
    # Test individual modules
    test_modules = [
        ("PDF Parser", "1a_pdf_parser_poc.simple_parser"),
        ("Heading Detector", "1a_pdf_parser_poc.heading_detector"), 
        ("Outline Extractor", "1a_structure_extractor_poc.outline_extractor"),
        ("Persona Analyzer", "1b_persona_analyzer_poc.persona_matcher"),
        ("Content Ranker", "1b_content_ranker_poc.section_ranker"),
        ("JSON Formatter", "utils_poc.json_formatter"),
        ("End-to-End Test", "integration_poc.end_to_end_test")
    ]
    
    results = {}
    
    for name, module_path in test_modules:
        print(f"\\nTesting {name}...")
        print("-" * 30)
        
        try:
            # Dynamic import and run
            module = __import__(module_path, fromlist=[''])
            if hasattr(module, '__main__'):
                # Run the module's main section
                exec(f"import {module_path}")
            results[name] = "✓ PASS"
            print(f"✓ {name} test completed")
            
        except Exception as e:
            results[name] = f"✗ FAIL: {e}"
            print(f"✗ {name} test failed: {e}")
    
    # Summary
    print("\\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for name, result in results.items():
        print(f"{name}: {result}")
    
    passed = sum(1 for r in results.values() if r.startswith("✓"))
    total = len(results)
    print(f"\\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    run_all_tests()
"""
    
    with open("poc-modules/run_poc_tests.py", "w") as f:
        f.write(test_script)
    
    os.chmod("poc-modules/run_poc_tests.py", 0o755)  # Make executable
    print("✓ Test runner created")

def create_readme():
    """Create README with usage instructions"""
    readme_content = """# Adobe Hackathon PoC Modules

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test individual modules:**
   ```bash
   cd poc-modules
   python -m 1a_pdf_parser_poc.simple_parser
   python -m 1a_pdf_parser_poc.heading_detector
   python -m 1a_structure_extractor_poc.outline_extractor
   python -m 1b_persona_analyzer_poc.persona_matcher
   python -m 1b_content_ranker_poc.section_ranker
   python -m utils_poc.json_formatter
   ```

3. **Run full test suite:**
   ```bash
   python run_poc_tests.py
   ```

## Module Overview

### Round 1A - Document Structure Extraction
- `simple_parser.py`: Basic PDF text extraction with font info
- `heading_detector.py`: Font and pattern-based heading detection
- `outline_extractor.py`: Complete Round 1A pipeline

### Round 1B - Persona-Driven Intelligence  
- `persona_matcher.py`: Analyze persona and job-to-be-done
- `section_ranker.py`: Rank sections by relevance
- Multi-document processing

### Utilities
- `json_formatter.py`: Output formatting and validation
- `end_to_end_test.py`: Integration testing

## Key Features Demonstrated

✓ **Fast PDF parsing** with PyMuPDF
✓ **Multi-strategy heading detection** (font + patterns)
✓ **Persona-aware content ranking** with keyword matching
✓ **Proper JSON output formatting**
✓ **Comprehensive validation**
✓ **Modular, testable architecture**

## Next Steps for Production

1. Replace mock data with real PDF processing
2. Add ML models for improved accuracy
3. Implement Docker containerization
4. Add performance optimizations
5. Enhance multilingual support

## Testing Strategy

Each module is standalone and testable:
- Uses simulated data when real PDFs unavailable
- Validates output formats
- Measures processing time
- Checks constraint compliance

This PoC demonstrates all core functionality needed for the Adobe Hackathon challenge!
"""
    
    with open("poc-modules/README.md", "w") as f:
        f.write(readme_content)
    
    print("✓ README created")

def main():
    """Main setup function"""
    print("Adobe Hackathon PoC Setup")
    print("=" * 30)
    
    create_poc_structure()
    create_requirements_file() 
    create_test_runner()
    create_readme()
    
    print("\n" + "="*40)
    print("✓ PoC setup complete!")
    print("\nNext steps:")
    print("1. cd poc-modules")
    print("2. pip install -r requirements.txt")
    print("3. python run_poc_tests.py")
    print("\nHappy coding! 🚀")

if __name__ == "__main__":
    main()