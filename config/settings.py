# config/settings.py

import os
from datetime import timedelta


# OpenAI API Key
API_KEY = os.getenv('OPENAI_API_KEY')

# Audio Configuration
SAMPLE_RATE = 44100  # 44.1kHz
CHANNELS = 1
AUDIO_DEVICE_INDEX = 2  # Adjust as needed

# Audio Processing Thresholds
AUDIO_THRESHOLD = 0.009  # Adjust based on your observations
SILENCE_DURATION_THRESHOLD = 1.0  # seconds
MIN_RECORDING_DURATION = 3.0  # seconds
MAX_RECORDING_DURATION = 30.0  # seconds
PRE_ROLL_DURATION = 1.0  # seconds
POST_ROLL_DURATION = 1.0  # seconds

# Queue Configuration
QUEUE_MAX_SIZE = 100  # Maximum number of audio clips to queue
TRANSCRIPTION_WORKERS = 4  # Number of threads for transcription

# Directories
TRANSCRIPTIONS_DIR = 'data/transcriptions'
AUDIO_DIR = 'data/audio'  # Directory to save audio files in debug mode
LOG_FILE = 'logs/dispatch_ai.log'

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

## Additional configurations for tx_stable.py

# Audio Configuration for Transmitter
TX_SAMPLE_RATE = 24000  # Adjust as needed
TX_CHANNELS = 1
TX_AUDIO_DEVICE_INDEX = 1  # Onboard headphone jack
AUDIO_DEVICE = "plughw:1,0"  # ALSA name for the onboard headphone jack

# File Paths
PROCESSED_FILES_JSON = 'data/processed_files.json'

# Directory to move processed transcription files
PROCESSED_TRANSCRIPTIONS_DIR = 'data/transcriptions/processed'

# Voice Configuration
VOICE_NAME = 'nova'  # Adjust as needed for TTS

# Directories
TTS_AUDIO_DIR = 'data/tts_audio'  # Directory to save TTS audio files in debug mode

# Conversation Settings
CONTEXT_EXPIRATION = timedelta(hours=1)
RESPONSE_QUEUE_MAX_SIZE = 100
CONVERSATION_HISTORY_LIMIT = 20  # Max number of messages to keep in history

# Logging Configuration for Transmitter
TX_LOG_FILE = 'logs/dispatch_ai_tx.log'

# Other Configurations
TRANSCRIPTIONS_DIR = 'data/transcriptions'
