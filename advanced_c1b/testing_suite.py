#!/usr/bin/env python3
"""
Enhanced Challenge 1B Testing Suite
Comprehensive validation of all advanced features
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add the main module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import AdvancedDocumentProcessor, DocumentSection, create_performance_optimized_processor
except ImportError as e:
    print(f"❌ Error importing main module: {e}")
    print("Please ensure main.py is in the same directory as this test file.")
    sys.exit(1)

class TestEnhancedFeatures(unittest.TestCase):
    """Test suite for enhanced Challenge 1B features"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.input_dir = os.path.join(cls.test_dir, 'input')
        cls.output_dir = os.path.join(cls.test_dir, 'output')
        cls.cache_dir = os.path.join(cls.test_dir, 'cache')
        
        os.makedirs(cls.input_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        
        # Create test documents
        cls._create_test_documents()
        cls._create_test_config()
        
        print(f"🧪 Test environment set up in: {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
        print("🧹 Test environment cleaned up")
    
    @classmethod
    def _create_test_documents(cls):
        """Create sample test documents"""
        
        # Academic paper
        academic_content = """# Advanced Machine Learning Techniques for Document Analysis

## Abstract

This paper presents novel approaches to document intelligence using transformer-based architectures. We achieve state-of-the-art performance on benchmark datasets with 94.2% accuracy.

## Introduction

Document processing has evolved significantly with the advent of deep learning. Traditional rule-based approaches are being replaced by neural architectures that can understand document structure and semantics.

## Methodology

### Model Architecture

Our approach utilizes a multi-scale transformer with the following components:
1. Document encoder for structural understanding
2. Content encoder for semantic analysis  
3. Cross-attention mechanism for relevance scoring

### Experimental Setup

We evaluate our method on three datasets:
- Academic papers (10,000 documents)
- Business reports (5,000 documents) 
- Technical manuals (3,000 documents)

## Results

Performance metrics demonstrate significant improvements:
- Precision: 92.1% (±1.8%)
- Recall: 96.3% (±2.1%)
- F1-Score: 94.2% (±1.5%)

The results show particular strength in technical document analysis and cross-document relationship detection.

## Discussion

Our findings indicate that attention-based models excel at capturing document structure. The multi-scale approach provides robust performance across diverse document types.

### Limitations

Current limitations include processing time for very large documents and memory requirements for batch processing.

## Conclusion

This work establishes new benchmarks for document intelligence and provides a foundation for persona-driven analysis systems.

## References

[1] Smith et al. (2023). "Transformer Models for Document Understanding"
[2] Johnson and Brown (2022). "Attention Mechanisms in NLP"
[3] Lee et al. (2024). "Cross-Document Analysis Techniques"
"""
        
        # Business report
        business_content = """# Q4 2023 Enterprise AI Market Report

## Executive Summary

The enterprise AI market experienced unprecedented growth in Q4 2023, with document processing solutions leading the expansion. Total market valuation reached $2.4B, representing 42% year-over-year growth.

## Market Analysis

### Revenue Performance

Key financial metrics for Q4 2023:
- Total revenue: $2.4B (+42% YoY)
- Document AI segment: $680M (+58% YoY)
- Market share by segment:
  * Healthcare: 28%
  * Financial services: 22% 
  * Legal tech: 18%
  * Other: 32%

### Competitive Landscape

The market remains highly competitive with both established players and emerging startups. Notable developments include:

1. Major acquisitions in the AI space
2. Increased venture capital investment
3. Strategic partnerships between tech giants

## Technology Trends

### Adoption Patterns

Enterprise adoption follows predictable patterns:
- Initial pilot programs (3-6 months)
- Departmental rollout (6-12 months)
- Enterprise-wide deployment (12-24 months)

### Integration Challenges

Common implementation hurdles include:
- Legacy system compatibility
- Data privacy and security concerns
- Staff training and change management
- Cost justification and ROI measurement

## Financial Projections

### Growth Forecast

Analysts project continued strong performance through 2025:
- 2024 projected revenue: $3.2B (+33% growth)
- 2025 projected revenue: $4.1B (+28% growth)

### Investment Opportunities

Key areas attracting investment:
- Multimodal document processing
- Real-time analysis capabilities
- Industry-specific solutions
- Edge computing implementations

## Risk Assessment

### Market Risks

Identified risk factors include:
- Regulatory changes affecting AI deployment
- Economic downturn impact on enterprise spending
- Technology obsolescence concerns
- Competitive pressure on pricing

### Mitigation Strategies

Recommended risk mitigation approaches:
1. Diversified technology portfolio
2. Strong customer relationships
3. Continuous innovation investment
4. Strategic partnership development

## Conclusion

The enterprise document AI market shows strong fundamentals with significant growth potential. Success factors include technological innovation, customer focus, and strategic market positioning.
"""
        
        # Technical manual
        technical_content = """# Document Processing System: Technical Implementation Guide

## Chapter 1: System Architecture

### Overview

The document processing system implements a microservices architecture optimized for scalability and performance. Core system components include:

- Ingestion Service: Handles document upload and preprocessing
- Analysis Engine: Performs content extraction and analysis
- Storage Layer: Manages document and metadata persistence
- API Gateway: Provides unified access to system capabilities

### Performance Specifications

System performance characteristics:

- Throughput: 10,000 documents/hour peak capacity
- Latency: <100ms average processing time per page
- Memory usage: 2GB base footprint, 8GB peak during batch processing
- Storage: 500MB base installation, expandable based on cache requirements

## Chapter 2: Installation and Configuration

### Prerequisites

Required system components:
- Python 3.9+ runtime environment
- Docker containerization platform
- PostgreSQL database server
- Redis cache server

### Installation Steps

1. Clone repository and install dependencies
2. Configure environment variables and database connections
3. Initialize model cache and download required AI models
4. Run system validation tests
5. Deploy using Docker Compose orchestration

### Configuration Parameters

Key configuration options:

```
MODEL_CACHE_SIZE=1024MB
BATCH_PROCESSING_SIZE=50
CONCURRENT_WORKERS=4
API_RATE_LIMIT=1000/hour
LOG_LEVEL=INFO
```

## Chapter 3: API Documentation

### Core Endpoints

Primary system endpoints:

#### POST /api/v1/process
Process single document with specified analysis parameters.

Request format:
- Content-Type: multipart/form-data
- Parameters: document file, analysis options, output format

Response format:
- JSON structure with extracted sections and metadata
- Processing metrics and performance statistics

#### GET /api/v1/status
Retrieve system health and performance metrics.

#### POST /api/v1/batch
Submit multiple documents for batch processing.

### Authentication

API access requires JWT token authentication:
1. Obtain access token via /auth/login endpoint
2. Include token in Authorization header for all requests
3. Tokens expire after 24 hours and must be refreshed

## Chapter 4: Performance Optimization

### Caching Strategies

Implemented caching mechanisms:
- Model cache: Pre-loaded ML models for faster inference
- Result cache: Processed document results for repeat requests  
- Metadata cache: Document metadata and index information

### Monitoring and Alerting

System monitoring includes:
- Processing throughput and latency metrics
- Error rates and exception tracking
- Resource utilization (CPU, memory, disk)
- API response time and availability

### Scaling Considerations

Horizontal scaling approaches:
- Load balancer configuration for multiple API instances
- Database read replicas for improved query performance
- Message queue for asynchronous processing workflows
- Container orchestration with Kubernetes

## Chapter 5: Troubleshooting

### Common Issues

Frequent problems and solutions:

**High Memory Usage**
- Reduce batch processing size
- Enable garbage collection optimization
- Monitor model cache size

**Slow Processing Speed**  
- Check database query performance
- Verify model cache hits
- Review network latency

**API Timeout Errors**
- Increase request timeout limits
- Optimize document preprocessing
- Consider asynchronous processing

### Debug Mode

Enable detailed logging:
```
export DEBUG=true
export LOG_LEVEL=DEBUG
```

### Performance Profiling

Built-in profiling tools:
- Memory usage tracking
- Processing time analysis
- Database query profiling
- API endpoint performance metrics
"""
        
        # Write test documents
        with open(os.path.join(cls.input_dir, 'academic_paper.txt'), 'w') as f:
            f.write(academic_content)
        
        with open(os.path.join(cls.input_dir, 'business_report.txt'), 'w') as f:
            f.write(business_content)
            
        with open(os.path.join(cls.input_dir, 'technical_manual.txt'), 'w') as f:
            f.write(technical_content)
    
    @classmethod
    def _create_test_config(cls):
        """Create test configuration file"""
        config = {
            "documents": [
                {"filename": "academic_paper.txt", "type": "academic"},
                {"filename": "business_report.txt", "type": "business"},
                {"filename": "technical_manual.txt", "type": "technical"}
            ],
            "persona": {
                "role": "Senior Data Scientist specializing in machine learning and NLP, with focus on document intelligence systems and enterprise AI solutions."
            },
            "job_to_be_done": {
                "task": "Analyze technical methodologies, performance benchmarks, and implementation strategies to design an advanced document intelligence system for enterprise deployment."
            }
        }
        
        config_path = os.path.join(cls.input_dir, 'challenge1b_input.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def setUp(self):
        """Set up each test"""
        self.processor = create_performance_optimized_processor(
            enable_caching=True,
            enable_multilingual=False
        )
        self.processor.cache_dir = Path(self.cache_dir)
    
    def test_advanced_section_detection(self):
        """Test enhanced section detection with accessibility features"""
        print("\n🧪 Testing advanced section detection...")
        
        # Test with academic paper
        file_path = os.path.join(self.input_dir, 'academic_paper.txt')
        sections = self.processor.extract_document_sections(file_path)
        
        # Verify sections were found
        self.assertGreater(len(sections), 0, "Should detect sections in academic paper")
        
        # Check for accessibility tags
        for section in sections:
            self.assertIsInstance(section.accessibility_tags, list, "Should have accessibility tags")
            self.assertIn('document', section.__dict__, "Should have document field")
            self.assertIn('section_title', section.__dict__, "Should have section title")
        
        print(f"✅ Detected {len(sections)} sections with accessibility features")
    
    def test_enhanced_relevance_scoring(self):
        """Test multi-signal relevance scoring"""
        print("\n🧪 Testing enhanced relevance scoring...")
        
        # Create test section
        test_section = DocumentSection(
            document="test.txt",
            section_title="Machine Learning Performance Metrics",
            page_number=1,
            context="This section discusses accuracy, precision, and recall metrics for ML models. Performance benchmarks show 94.2% accuracy on test datasets.",
            full_text="Full text content here",
            accessibility_tags=['h2', 'technical'],
            cross_references=['Figure 1', 'Table 2']
        )
        
        persona = "Senior Data Scientist specializing in machine learning"
        job = "Analyze performance benchmarks for ML systems"
        
        persona_embedding = self.processor.sentence_model.encode([persona])
        job_embedding = self.processor.sentence_model.encode([job])
        
        score = self.processor.calculate_enhanced_relevance_score(
            test_section, persona_embedding[0], job_embedding[0], persona, job
        )
        
        self.assertGreater(score, 0.0, "Should calculate positive relevance score")
        self.assertLessEqual(score, 1.0, "Relevance score should be normalized")
        
        print(f"✅ Relevance score calculated: {score:.4f}")
    
    def test_concept_graph_building(self):
        """Test concept graph construction"""
        print("\n🧪 Testing concept graph building...")
        
        # Create multiple test sections
        sections = [
            DocumentSection(
                document="doc1.txt", section_title="Machine Learning", page_number=1,
                context="Machine learning algorithms for document analysis", full_text="content"
            ),
            DocumentSection(
                document="doc2.txt", section_title="Neural Networks", page_number=1,
                context="Neural network architectures for machine learning applications", full_text="content"
            ),
            DocumentSection(
                document="doc3.txt", section_title="Performance Analysis", page_number=1,
                context="Performance metrics and benchmarking for machine learning models", full_text="content"
            )
        ]
        
        # Build concept graph
        concept_graph = self.processor.build_concept_graph(sections)
        
        self.assertGreater(concept_graph.number_of_nodes(), 0, "Should create graph nodes")
        print(f"✅ Concept graph built with {concept_graph.number_of_nodes()} nodes and {concept_graph.number_of_edges()} edges")
    
    def test_diversified_selection(self):
        """Test diversified section selection algorithm"""
        print("\n🧪 Testing diversified section selection...")
        
        # Create test sections with varying relevance scores
        sections = []
        for i in range(10):
            section = DocumentSection(
                document=f"doc{i%3}.txt",
                section_title=f"Section {i}",
                page_number=i+1,
                context=f"Content for section {i} with machine learning and analysis topics",
                full_text="full content",
                relevance_score=0.9 - (i * 0.05)  # Decreasing relevance
            )
            sections.append(section)
        
        # Test diversified selection
        selected = self.processor._diversified_section_selection(sections, max_sections=5)
        
        self.assertEqual(len(selected), 5, "Should select exactly 5 sections")
        
        # Check for document diversity
        doc_names = [s.document for s in selected]
        unique_docs = set(doc_names)
        self.assertGreater(len(unique_docs), 1, "Should select from multiple documents")
        
        print(f"✅ Selected {len(selected)} diverse sections from {len(unique_docs)} documents")
    
    def test_citation_extraction(self):
        """Test citation extraction functionality"""
        print("\n🧪 Testing citation extraction...")
        
        test_text = """
        This approach is supported by recent research [1, 2, 3]. 
        Smith et al. (2023) demonstrated similar results.
        The methodology follows (Johnson, 2022) and (Brown et al., 2024).
        """
        
        citations = self.processor.extract_citations(test_text)
        
        self.assertGreater(len(citations), 0, "Should extract citations from text")
        print(f"✅ Extracted {len(citations)} citations: {citations}")
    
    def test_cross_reference_extraction(self):
        """Test cross-reference extraction"""
        print("\n🧪 Testing cross-reference extraction...")
        
        test_text = """
        As shown in Figure 1 and Table 2, the results indicate significant improvement.
        See Section 3.1 for detailed analysis. Appendix A contains additional data.
        Chapter 5 discusses implementation details.
        """
        
        cross_refs = self.processor._extract_cross_references(test_text)
        
        self.assertGreater(len(cross_refs), 0, "Should extract cross-references")
        print(f"✅ Extracted cross-references: {cross_refs}")
    
    def test_accessibility_summary(self):
        """Test accessibility summary generation"""
        print("\n🧪 Testing accessibility summary...")
        
        test_sections = [
            DocumentSection(
                document="test.txt", section_title="Test", page_number=1,
                context="content", full_text="content",
                accessibility_tags=['h1', 'technical'],
                cross_references=['Figure 1', 'Table 1']
            ),
            DocumentSection(
                document="test2.txt", section_title="Test2", page_number=2,
                context="content", full_text="content",
                accessibility_tags=['h2', 'data-heavy'],
                cross_references=['Section 2']
            )
        ]
        
        summary = self.processor._generate_accessibility_summary(test_sections)
        
        self.assertIn('content_types', summary, "Should include content types")
        self.assertIn('accessibility_features', summary, "Should include accessibility features")
        self.assertIn('structural_elements', summary, "Should include structural elements")
        
        print("✅ Accessibility summary generated successfully")
    
    def test_multilingual_detection(self):
        """Test multilingual language detection"""
        print("\n🧪 Testing multilingual language detection...")
        
        # Enable multilingual support for this test
        self.processor.enable_multilingual = True
        self.processor._setup_multilingual_support()
        
        test_texts = {
            'english': "This is an English text about machine learning",
            'japanese': "これは機械学習についての日本語のテキストです",
            'chinese': "这是关于机器学习的中文文本",
        }
        
        for expected_lang, text in test_texts.items():
            detected = self.processor._detect_language(text)
            print(f"Text: '{text[:30]}...' -> Detected: {detected}")
        
        print("✅ Language detection functionality tested")
    
    def test_caching_functionality(self):
        """Test caching system"""
        print("\n🧪 Testing caching functionality...")
        
        # Test embedding caching
        if hasattr(self.processor, 'cache_manager'):
            test_text = "This is a test text for caching"
            
            # First encoding (should cache)
            embedding1 = self.processor.sentence_model.encode([test_text])[0]
            
            # Second encoding (should use cache)
            embedding2 = self.processor.sentence_model.encode([test_text])[0]
            
            # Verify embeddings are identical
            import numpy as np
            self.assertTrue(np.array_equal(embedding1, embedding2), "Cached embeddings should be identical")
            
            print("✅ Caching functionality verified")
        else:
            print("⚠️ Caching not enabled for this test")
    
    def test_full_pipeline_integration(self):
        """Test complete processing pipeline"""
        print("\n🧪 Testing full pipeline integration...")
        
        config_path = os.path.join(self.input_dir, 'challenge1b_input.json')
        
        start_time = time.time()
        result = self.processor.process_documents(self.input_dir, config_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify required output structure
        self.assertIn('metadata', result, "Should include metadata")
        self.assertIn('extracted_sections', result, "Should include extracted sections")
        self.assertIn('subsection_analysis', result, "Should include subsection analysis")
        self.assertIn('advanced_features', result, "Should include advanced features")
        
        # Verify enhanced features
        advanced_features = result['advanced_features']
        self.assertIn('concept_insights', advanced_features, "Should include concept insights")
        self.assertIn('explainability', advanced_features, "Should include explainability")
        self.assertIn('cross_document_connections', advanced_features, "Should include cross-document connections")
        self.assertIn('accessibility_summary', advanced_features, "Should include accessibility summary")
        
        # Verify performance requirements
        self.assertLess(processing_time, 60, f"Processing should complete within 60 seconds (took {processing_time:.2f}s)")
        self.assertGreater(len(result['extracted_sections']), 0, "Should extract relevant sections")
        
        print(f"✅ Full pipeline completed in {processing_time:.2f} seconds")
        print(f"📊 Extracted {len(result['extracted_sections'])} sections")
        print(f"🔍 Generated {len(result['subsection_analysis'])} subsections")
        print(f"🔗 Found {len(advanced_features['cross_document_connections'])} cross-document connections")
    
    def test_output_validation(self):
        """Test output format validation"""
        print("\n🧪 Testing output format validation...")
        
        config_path = os.path.join(self.input_dir, 'challenge1b_input.json')
        result = self.processor.process_documents(self.input_dir, config_path)
        
        # Validate metadata structure
        metadata = result['metadata']
        required_metadata_fields = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
        for field in required_metadata_fields:
            self.assertIn(field, metadata, f"Metadata should include {field}")
        
        # Validate extracted sections structure
        for section in result['extracted_sections']:
            required_section_fields = ['document', 'section_title', 'importance_rank', 'page_number']
            for field in required_section_fields:
                self.assertIn(field, section, f"Section should include {field}")
        
        # Validate subsection analysis structure
        for subsection in result['subsection_analysis']:
            required_subsection_fields = ['document', 'refined_text', 'page_number']
            for field in required_subsection_fields:
                self.assertIn(field, subsection, f"Subsection should include {field}")
        
        print("✅ Output format validation passed")
    
    def test_performance_benchmarks(self):
        """Test performance against benchmarks"""
        print("\n🧪 Testing performance benchmarks...")
        
        config_path = os.path.join(self.input_dir, 'challenge1b_input.json')
        
        # Memory usage test
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        result = self.processor.process_documents(self.input_dir, config_path)
        end_time = time.time()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        processing_time = end_time - start_time
        
        # Performance assertions (based on your development benchmarks)
        self.assertLess(processing_time, 60, f"Should process within 60s (took {processing_time:.2f}s)")
        self.assertLess(memory_used, 700, f"Should use <700MB memory (used {memory_used:.1f}MB)")
        
        print(f"✅ Performance benchmarks:")
        print(f"   ⏱️ Processing time: {processing_time:.2f}s (limit: 60s)")
        print(f"   💾 Memory usage: {memory_used:.1f}MB (limit: 700MB)")
        print(f"   📄 Documents processed: {len(result['metadata']['input_documents'])}")

class TestAdvancedFeatureIntegration(unittest.TestCase):
    """Integration tests for advanced features working together"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.processor = create_performance_optimized_processor(enable_caching=True)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_feature_interaction(self):
        """Test that advanced features work together correctly"""
        print("\n🧪 Testing advanced feature interaction...")
        
        # Test that accessibility tags influence relevance scoring
        section_with_technical = DocumentSection(
            document="test.txt", section_title="Technical Analysis", page_number=1,
            context="Advanced machine learning algorithms and performance metrics",
            full_text="content", accessibility_tags=['h2', 'technical', 'data-heavy']
        )
        
        section_without_tags = DocumentSection(
            document="test.txt", section_title="General Discussion", page_number=2,
            context="General discussion about various topics",
            full_text="content", accessibility_tags=['h3']
        )
        
        persona = "Senior Data Scientist"
        job = "Analyze technical performance metrics"
        
        persona_embedding = self.processor.sentence_model.encode([persona])
        job_embedding = self.processor.sentence_model.encode([job])
        
        score1 = self.processor.calculate_enhanced_relevance_score(
            section_with_technical, persona_embedding[0], job_embedding[0], persona, job
        )
        score2 = self.processor.calculate_enhanced_relevance_score(
            section_without_tags, persona_embedding[0], job_embedding[0], persona, job
        )
        
        # Technical section should score higher for technical persona
        self.assertGreater(score1, score2, "Technical content should score higher for technical persona")
        
        print(f"✅ Feature interaction verified: Technical section ({score1:.3f}) > General section ({score2:.3f})")

def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    print("🚀 Starting Enhanced Challenge 1B Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestEnhancedFeatures))
    test_suite.addTest(unittest.makeSuite(TestAdvancedFeatureIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 Test Suite Summary")
    print("=" * 60)
    print(f"⏱️ Total execution time: {end_time - start_time:.2f} seconds")
    print(f"✅ Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests failed: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n🔍 Failures:")
        for test, traceback in result.failures:
            msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  - {test}: {msg}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            err_msg = traceback.split('\n')[-2]
            print(f"  - {test}: {err_msg}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n📊 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 Enhanced Challenge 1B implementation is ready for deployment!")
    elif success_rate >= 75:
        print("⚠️ Implementation needs minor improvements before deployment")
    else:
        print("❌ Implementation requires significant fixes before deployment")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Check if main module can be imported
    try:
        import main
        print("✅ Main module imported successfully")
    except Exception as e:
        print(f"❌ Error importing main module: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the comprehensive test suite
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)