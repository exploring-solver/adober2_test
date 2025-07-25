import os
from pathlib import Path

# Base Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
JSON_OUTPUT_DIR = OUTPUT_DIR / "json"
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
DEBUG_OUTPUT_DIR = OUTPUT_DIR / "debug"
MODEL_DIR = DATA_DIR / "models"

# Processing Configuration
MAX_PROCESSING_TIME = 10  # seconds
BATCH_SIZE = 32
MAX_FILE_SIZE_MB = 100

# Font Analysis Thresholds
# Make font thresholds more inclusive
FONT_SIZE_THRESHOLD_RATIO = 1.05  # Only 5% larger than average
MIN_FONT_SIZE_FOR_HEADING = 10    # Absolute minimum
RELATIVE_SIZE_BONUS = 1.1         # Even small increases count
BOLD_WEIGHT_THRESHOLD = 600
MIN_HEADING_LENGTH = 3
MAX_HEADING_LENGTH = 200

# Semantic Filtering
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_SIMILARITY_THRESHOLD = 0.3
CONTEXT_WINDOW = 2  # paragraphs before/after

# Hierarchy Assignment
MAX_HIERARCHY_LEVELS = 6
TITLE_POSITION_THRESHOLD = 0.1  # top 10% of page
CENTER_ALIGNMENT_TOLERANCE = 0.1

# Output Configuration
OUTPUT_FORMAT = "json"
INCLUDE_CONFIDENCE_SCORES = True
INCLUDE_DEBUG_INFO = False