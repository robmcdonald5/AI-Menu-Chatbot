import logging
from logging.handlers import RotatingFileHandler

# ------------------------------
# Debug Configuration
# ------------------------------
DEBUG = True  # Set to False in production

# ------------------------------
# CORS Configuration
# ------------------------------
CORS_ORIGINS = [
    #"http://localhost:5001",
    "https://chipotleaimenu.app"  # Uncomment this line for production
]

# ------------------------------
# Logging Configuration
# ------------------------------
LOG_FILE = "chatbot.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5MB per log file
LOG_BACKUP_COUNT = 5  # Number of backup log files

# Logging Levels
ROOT_LOG_LEVEL = logging.INFO
CONSOLE_LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO

# Logging Format
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# External Libraries Logging Suppression
EXTERNAL_LOGGERS = [
    'pymongo',
    'urllib3',
    'sentence_transformers',
    'spacy',
    'werkzeug'
]

# ------------------------------
# Similarity Configuration
# ------------------------------
SIMILARITY_THRESHOLD_HIGH = 0.7  # High confidence
SIMILARITY_THRESHOLD_MEDIUM = 0.45  # Medium confidence

# Similarity Weights
WEIGHT_COSINE = 0.5
WEIGHT_EUCLIDEAN = 0.3
WEIGHT_JACCARD = 0.2

# ------------------------------
# Field Prompts for Slot-Filling
# ------------------------------
FIELD_PROMPTS = {
    "meats": "What kind of meat would you like?",
    "rice": "What type of rice would you like?",
    "beans": "What type of beans would you like?",
    "toppings": "What toppings would you like?"
}

# ------------------------------
# Bot Configuration
# ------------------------------
BOT_NAME = "Chipotle"

# ------------------------------
# Inactivity Timeout
# ------------------------------
INACTIVITY_TIMEOUT_MINUTES = 5

# ------------------------------
# Model Configuration
# ------------------------------
SPACY_MODEL = 'en_core_web_sm'
SENTENCE_MODEL = 'all-mpnet-base-v2'

# ------------------------------
# Intents Configuration
# ------------------------------
INTENTS_FILE = 'intents.json'