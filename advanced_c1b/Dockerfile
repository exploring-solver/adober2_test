# Enhanced Challenge 1B Dockerfile with Advanced Features
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for enhanced features
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create directories for enhanced functionality
RUN mkdir -p /app/input /app/output /app/cache /app/model_cache

# Copy requirements for enhanced features
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download and cache models for faster startup
RUN python -c "import spacy; import nltk; from sentence_transformers import SentenceTransformer; from transformers import pipeline; \
spacy.cli.download('en_core_web_sm'); \
model = SentenceTransformer('all-MiniLM-L6-v2'); print('Sentence transformer model cached'); \
nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); print('NLTK data downloaded')"


# Copy enhanced application code
COPY main.py .
COPY approach_explanation.md .

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Create startup script for enhanced features
RUN echo '#!/bin/bash\n\
echo "🚀 Starting Enhanced Challenge 1B Processing..."\n\
echo "💾 Cache directory: /app/cache"\n\
echo "🤖 Model cache: /app/model_cache"\n\
echo "📁 Input directory: /app/input"\n\
echo "📤 Output directory: /app/output"\n\
echo ""\n\
# Check available features\n\
echo "🔧 Available Enhanced Features:"\n\
echo "  ✅ Advanced Section Detection with Accessibility"\n\
echo "  ✅ Multi-Signal Relevance Scoring"\n\
echo "  ✅ Cross-Document Concept Graph"\n\
echo "  ✅ Citation-Aware Ranking"\n\
echo "  ✅ Explainability Reports"\n\
echo "  ✅ Performance Optimizations"\n\
echo "  ✅ Intelligent Caching"\n\
if [ "$ENABLE_MULTILINGUAL" = "true" ]; then\n\
    echo "  ✅ Multilingual Support Enabled"\n\
fi\n\
echo ""\n\
# Run the enhanced processor\n\
python main.py \\\n\
    --input_dir /app/input \\\n\
    --output_dir /app/output \\\n\
    --input_file challenge1b_input.json \\\n\
    --output_file challenge1b_output.json \\\n\
    --enable_caching \\\n\
    ${ENABLE_MULTILINGUAL:+--enable_multilingual} \\\n\
    ${DEBUG:+--debug}\n\
echo ""\n\
echo "✅ Enhanced processing complete!"\n\
echo "📊 Check /app/output/performance_report.json for detailed metrics"\n\
echo "🔍 See /app/output/challenge1b_output.json for results"\n\
' > /app/start_enhanced.sh && chmod +x /app/start_enhanced.sh

# Create performance monitoring script
RUN echo '#!/bin/bash\n\
echo "📈 Performance Monitoring Active..."\n\
echo "Memory usage: $(free -h | grep Mem | awk '"'"'{print $3"/"$2}'"'"')"\n\
echo "CPU cores: $(nproc)"\n\
echo "Python version: $(python --version)"\n\
echo "Cache size: $(du -sh /app/cache 2>/dev/null || echo "0B")"\n\
echo "Model cache size: $(du -sh /app/model_cache 2>/dev/null || echo "0B")"\n\
' > /app/monitor.sh && chmod +x /app/monitor.sh

# Copy health check script
COPY health_check.py .

# Health check for enhanced features
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python /app/health_check.py

# Set proper permissions
RUN chmod -R 755 /app/cache /app/model_cache /app/output

# Default command runs enhanced processor
CMD ["/app/start_enhanced.sh"]

# Labels for enhanced features
LABEL version="2.0-enhanced"
LABEL description="Challenge 1B Enhanced: Persona-Driven Document Intelligence with Advanced Features"
LABEL features="accessibility,concept-graph,citation-analysis,caching,explainability"
LABEL author="Enhanced Challenge 1B Team"
LABEL performance="<60s processing, <1GB models, <680MB memory"
