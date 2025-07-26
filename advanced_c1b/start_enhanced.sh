#!/bin/bash

# Enhanced Challenge 1B Deployment Script
# Demonstrates all advanced features and optimizations

set -e

echo "🚀 Enhanced Challenge 1B Deployment Script"
echo "============================================"
echo ""

# Configuration
CONTAINER_NAME="challenge1b-enhanced"
IMAGE_NAME="challenge1b:enhanced"
INPUT_DIR="./input"
OUTPUT_DIR="./output"
CACHE_DIR="./cache"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${BLUE}📋 Step: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
print_step "Checking prerequisites"
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_success "Docker is available"

# Create necessary directories
print_step "Setting up directory structure"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$CACHE_DIR"
print_success "Directory structure created"

# Check for existing container and remove if needed
print_step "Cleaning up existing containers"
if docker ps -a --format 'table {{.Names}}' | grep -q "$CONTAINER_NAME"; then
    print_warning "Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
fi

# Build enhanced Docker image
print_step "Building enhanced Docker image"
echo "Building with advanced features enabled..."

cat > Dockerfile.tmp << 'EOF'
# Enhanced Challenge 1B Dockerfile with Advanced Features
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl wget unzip git \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /app/input /app/output /app/cache /app/model_cache

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download models for performance
RUN python -c "
import spacy
from sentence_transformers import SentenceTransformer
spacy.cli.download('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Models cached successfully')
"

# Copy application
COPY main.py approach_explanation.md ./

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Create enhanced startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Enhanced Challenge 1B Processing Starting..."\n\
echo "Features: Accessibility, Concept Graph, Citation Analysis, Caching"\n\
python main.py \\\n\
    --input_dir /app/input \\\n\
    --output_dir /app/output \\\n\
    --input_file challenge1b_input.json \\\n\
    --output_file challenge1b_output.json \\\n\
    --enable_caching \\\n\
    ${ENABLE_MULTILINGUAL:+--enable_multilingual} \\\n\
    ${DEBUG:+--debug}\n\
echo "✅ Enhanced processing complete!"\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
EOF

docker build -f Dockerfile.tmp -t "$IMAGE_NAME" .
rm Dockerfile.tmp
print_success "Enhanced Docker image built: $IMAGE_NAME"

# Create sample input if not exists
print_step "Setting up sample test data"
if [ ! -f "$INPUT_DIR/challenge1b_input.json" ]; then
    cat > "$INPUT_DIR/challenge1b_input.json" << 'EOF'
{
  "documents": [
    {
      "filename": "research_paper_1.txt",
      "type": "academic"
    },
    {
      "filename": "business_report.txt", 
      "type": "business"
    },
    {
      "filename": "technical_manual.txt",
      "type": "technical"
    }
  ],
  "persona": {
    "role": "Senior Data Scientist with expertise in machine learning and natural language processing, focusing on practical applications in business intelligence and document analysis systems."
  },
  "job_to_be_done": {
    "task": "Extract and analyze the most relevant technical methodologies, performance benchmarks, and implementation strategies from the provided documents to inform the development of an enterprise document intelligence system."
  }
}
EOF

    # Create sample documents if they don't exist
    if [ ! -f "$INPUT_DIR/research_paper_1.txt" ]; then
        cat > "$INPUT_DIR/research_paper_1.txt" << 'EOF'
# Advanced Document Intelligence Systems: A Comprehensive Review

## Abstract

This paper presents a comprehensive review of document intelligence systems, focusing on recent advances in natural language processing and machine learning techniques. We analyze performance benchmarks across multiple datasets and propose novel methodologies for persona-driven document analysis.

## Introduction

Document intelligence has emerged as a critical component of modern information systems. Traditional approaches rely heavily on rule-based parsing, while contemporary methods leverage deep learning architectures for enhanced accuracy and adaptability.

## Methodology

### Neural Architecture Design

Our proposed system utilizes transformer-based models with attention mechanisms specifically designed for document structure understanding. The architecture incorporates:

1. Multi-scale feature extraction
2. Hierarchical attention layers  
3. Cross-document relationship modeling

### Performance Benchmarks

Experimental results demonstrate significant improvements over baseline methods:
- Accuracy: 94.2% (±2.1%)
- Processing speed: 15ms per document
- Memory efficiency: 45% reduction vs. previous models

## Results and Discussion

The evaluation across three diverse datasets shows consistent performance gains. Key findings include improved handling of technical documents and enhanced persona-specific relevance scoring.

## Conclusion

This work establishes new benchmarks for document intelligence systems and provides a foundation for future research in persona-driven analysis.
EOF
    fi

    if [ ! -f "$INPUT_DIR/business_report.txt" ]; then
        cat > "$INPUT_DIR/business_report.txt" << 'EOF'
# Enterprise AI Solutions: Market Analysis Report 2024

## Executive Summary

The enterprise AI market continues rapid expansion, with document processing solutions representing a key growth segment. Revenue projections indicate 40% year-over-year growth in the intelligent document processing sector.

## Market Overview

### Revenue Analysis

Q4 2023 performance metrics:
- Total market size: $2.4B
- Document AI segment: $680M (28% of total)
- Growth rate: 42% YoY

### Key Players

Leading vendors in the document intelligence space include established tech giants and emerging specialized providers. Market share distribution shows increasing fragmentation as new technologies emerge.

## Technology Trends

### Machine Learning Integration

Advanced ML techniques are becoming standard in enterprise deployments:
- Natural language processing integration
- Computer vision for document layout analysis
- Automated workflow optimization

### Implementation Strategies

Successful deployments typically follow phased approaches:
1. Pilot program with limited document types
2. Gradual expansion to additional use cases
3. Full enterprise integration with existing systems

## Financial Projections

Market analysts project continued strong growth through 2025, with particular strength in:
- Healthcare document processing
- Financial services automation
- Legal document analysis

## Risk Factors

Key challenges include data privacy concerns, integration complexity, and the need for specialized technical expertise in implementation teams.
EOF
    fi

    if [ ! -f "$INPUT_DIR/technical_manual.txt" ]; then
        cat > "$INPUT_DIR/technical_manual.txt" << 'EOF'
# Document Processing System Technical Manual

## Chapter 1: System Architecture

### Overview

The document processing system follows a modular architecture designed for scalability and maintainability. Core components include:

- Document ingestion layer
- Text extraction engine  
- Analysis pipeline
- Output formatting module

### Performance Specifications

System requirements and performance characteristics:

- Processing capacity: 10,000 documents/hour
- Memory footprint: <2GB RAM
- Storage requirements: 500MB base installation
- CPU utilization: Optimized for multi-core processing

## Chapter 2: Implementation Guide

### Installation Process

1. Download required dependencies
2. Configure environment variables
3. Initialize model cache
4. Verify system functionality

### Configuration Options

The system supports extensive configuration through environment variables and configuration files. Key parameters include:

- Model selection and optimization settings
- Cache management policies
- Output format specifications
- Logging and monitoring configuration

## Chapter 3: API Reference

### Core Functions

Primary API endpoints provide access to document processing capabilities:

- `/process`: Main document processing endpoint
- `/analyze`: Advanced analysis functions
- `/export`: Result export utilities

### Error Handling

Comprehensive error handling ensures robust operation:
- Input validation and sanitization
- Graceful degradation for unsupported formats
- Detailed error reporting and logging

## Chapter 4: Optimization Techniques

### Performance Tuning

Several optimization strategies can improve system performance:

- Model quantization for reduced memory usage
- Batch processing for improved throughput
- Caching strategies for repeated operations
- Parallel processing configuration

### Monitoring and Maintenance

Regular monitoring ensures optimal performance:
- Resource utilization tracking
- Processing time analysis
- Error rate monitoring
- Cache efficiency metrics
EOF
    fi

    print_success "Sample test data created in $INPUT_DIR"
else
    print_success "Input configuration found: $INPUT_DIR/challenge1b_input.json"
fi

# Function to run enhanced processing
run_enhanced_processing() {
    local mode=$1
    local extra_args=$2
    
    print_step "Running enhanced processing (Mode: $mode)"
    
    echo "🔧 Enhanced Features Active:"
    echo "  ✅ Advanced Section Detection with Accessibility Tagging"
    echo "  ✅ Multi-Signal Relevance Scoring (Semantic + TF-IDF + Citations)"
    echo "  ✅ Cross-Document Concept Graph Analysis"
    echo "  ✅ Citation-Aware Ranking System"
    echo "  ✅ Explainability and Transparency Reports"
    echo "  ✅ Intelligent Caching for Performance"
    echo "  ✅ Diversified Selection to Avoid Redundancy"
    echo ""
    
    # Run the enhanced container
    docker run --rm \
        --name "$CONTAINER_NAME" \
        -v "$(pwd)/$INPUT_DIR:/app/input:ro" \
        -v "$(pwd)/$OUTPUT_DIR:/app/output" \
        -v "$(pwd)/$CACHE_DIR:/app/cache" \
        -e ENABLE_MULTILINGUAL="${ENABLE_MULTILINGUAL:-false}" \
        -e DEBUG="${DEBUG:-false}" \
        $extra_args \
        "$IMAGE_NAME"
}

# Interactive menu
show_menu() {
    echo ""
    echo "🎯 Enhanced Challenge 1B Deployment Options"
    echo "==========================================="
    echo "1) Run Standard Enhanced Processing"
    echo "2) Run with Multilingual Support"
    echo "3) Run with Debug Mode"
    echo "4) Run Performance Benchmark"
    echo "5) Show System Information"
    echo "6) Clean Up Resources"
    echo "7) Exit"
    echo ""
}

# Performance benchmark
run_benchmark() {
    print_step "Running performance benchmark"
    
    echo "📊 Benchmarking enhanced features..."
    start_time=$(date +%s)
    
    # Run with timing
    docker run --rm \
        --name "$CONTAINER_NAME-benchmark" \
        -v "$(pwd)/$INPUT_DIR:/app/input:ro" \
        -v "$(pwd)/$OUTPUT_DIR:/app/output" \
        -v "$(pwd)/$CACHE_DIR:/app/cache" \
        -e DEBUG="true" \
        "$IMAGE_NAME" /bin/bash -c "
        echo '⏱️ Starting benchmark...'
        time python main.py \
            --input_dir /app/input \
            --output_dir /app/output \
            --input_file challenge1b_input.json \
            --output_file challenge1b_output.json \
            --enable_caching \
            --debug
        echo '📈 Benchmark complete!'
        "
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    print_success "Benchmark completed in ${duration} seconds"
    
    if [ -f "$OUTPUT_DIR/performance_report.json" ]; then
        echo ""
        echo "📊 Performance Report:"
        cat "$OUTPUT_DIR/performance_report.json" | python3 -m json.tool 2>/dev/null || cat "$OUTPUT_DIR/performance_report.json"
    fi
}

# Show system information
show_system_info() {
    print_step "System Information"
    
    echo "🖥️ Docker System Info:"
    docker system info --format "table {{.ServerVersion}}\t{{.Architecture}}\t{{.OSType}}"
    
    echo ""
    echo "💾 Available Resources:"
    docker system df
    
    echo ""
    echo "🏷️ Enhanced Image Details:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    if [ -d "$CACHE_DIR" ]; then
        echo ""
        echo "📁 Cache Directory:"
        du -sh "$CACHE_DIR" 2>/dev/null || echo "Cache directory empty"
    fi
}

# Clean up resources
cleanup_resources() {
    print_step "Cleaning up resources"
    
    # Stop any running containers
    docker ps -q --filter "name=$CONTAINER_NAME" | xargs -r docker stop
    
    # Remove containers
    docker ps -aq --filter "name=$CONTAINER_NAME" | xargs -r docker rm
    
    # Option to remove image
    read -p "Remove Docker image $IMAGE_NAME? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rmi "$IMAGE_NAME" 2>/dev/null || print_warning "Image not found or in use"
        print_success "Docker image removed"
    fi
    
    # Option to clear cache
    read -p "Clear cache directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$CACHE_DIR"/* 2>/dev/null || true
        print_success "Cache cleared"
    fi
    
    print_success "Cleanup completed"
}

# Main interactive loop
while true; do
    show_menu
    read -p "Select option [1-7]: " choice
    
    case $choice in
        1)
            run_enhanced_processing "standard" ""
            if [ $? -eq 0 ]; then
                print_success "Enhanced processing completed successfully!"
                echo ""
                echo "📄 Output files created:"
                ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No output files found"
            else
                print_error "Processing failed. Check logs above."
            fi
            ;;
        2)
            export ENABLE_MULTILINGUAL=true
            run_enhanced_processing "multilingual" ""
            unset ENABLE_MULTILINGUAL
            ;;
        3)
            export DEBUG=true
            run_enhanced_processing "debug" ""
            unset DEBUG
            ;;
        4)
            run_benchmark
            ;;
        5)
            show_system_info
            ;;
        6)
            cleanup_resources
            ;;
        7)
            print_success "Enhanced Challenge 1B deployment script completed!"
            echo "Thank you for using the enhanced document intelligence system! 🚀"
            exit 0
            ;;
        *)
            print_warning "Invalid option. Please select 1-7."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done