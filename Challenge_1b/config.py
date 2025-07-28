# config.py

# Model settings
MODEL_NAME = 'all-MiniLM-L6-v2'  # A fast, high-quality model under 1GB
RELEVANCE_THRESHOLD = 0.3         # Minimum score to be considered relevant

# Output settings
TOP_N_RESULTS = 15                # Number of top results to include in the final JSON

# Performance settings
MAX_WORKERS = 8                   # Max number of PDFs to process at the same time