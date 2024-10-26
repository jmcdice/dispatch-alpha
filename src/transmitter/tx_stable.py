# src/transmitter/tx_stable.py

import os
import sys
import shutil
import json
import time
import logging
from logging.handlers import WatchedFileHandler
import threading
from datetime import datetime, timedelta
import queue
import argparse
import warnings
import subprocess
import random
import requests

# Ignore DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import settings and utilities
from config.settings import (
    API_KEY,
    TX_SAMPLE_RATE,
    TX_CHANNELS,
    TX_AUDIO_DEVICE_INDEX,
    AUDIO_DEVICE,
    PROCESSED_FILES_JSON,
    VOICE_NAME,  # Default voice, will be overridden per persona
    TTS_AUDIO_DIR,
    CONTEXT_EXPIRATION,
    RESPONSE_QUEUE_MAX_SIZE,
    CONVERSATION_HISTORY_LIMIT,
    TX_LOG_FILE,
    LOG_FORMAT,
    TRANSCRIPTIONS_DIR,
    PROCESSED_TRANSCRIPTIONS_DIR,
    TRANSCRIPTIONS_LOG_FILE,
    TTS_PROVIDER,
    UNREALSPEECH_API_KEY,
    DEFAULT_VOICE,
    VOICE_MAPPING
)
from src.common.utils import initialize_logging, register_signal_handlers

from openai import OpenAI

# Initialize Logging
initialize_logging(TX_LOG_FILE, LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize Transcription Logger
transcription_logger = logging.getLogger('transcriptions')
transcription_logger.setLevel(logging.INFO)
transcription_handler = WatchedFileHandler(TRANSCRIPTIONS_LOG_FILE)
transcription_handler.setLevel(logging.INFO)
transcription_formatter = logging.Formatter('%(asctime)s | %(processName)s | assistant | %(message)s')
transcription_handler.setFormatter(transcription_formatter)
transcription_logger.addHandler(transcription_handler)

# Initialize OpenAI Client
if not API_KEY:
    logger.error("OPENAI_API_KEY is not set. Please set it as an environment variable.")
    sys.exit(1)

os.makedirs(PROCESSED_TRANSCRIPTIONS_DIR, exist_ok=True)

client = OpenAI(api_key=API_KEY)

# Global Variables
terminate_flag = threading.Event()
response_queue = queue.Queue(maxsize=RESPONSE_QUEUE_MAX_SIZE)
conversation_history = []
debug_mode = False  # Will be set based on command-line arguments
personas = {}  # Dictionary to store loaded personas
activation_phrases_set = set()  # Set of all activation phrases to avoid duplicates

active_persona = None  # Currently active persona
last_interaction_time = None  # Timestamp of the last interaction
CONVERSATION_TIMEOUT = timedelta(minutes=5)  # Adjust the timeout as needed

assistant_responses = []  # List to store recent assistant responses

LOCK_FILE = '/tmp/tx_rx_lock'

def create_lock():
    """Create a lock file to signal the receiver to pause."""
    with open(LOCK_FILE, 'w') as f:
        f.write('locked')

def remove_lock():
    """Remove the lock file to signal the receiver to resume."""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

# Register Signal Handlers
register_signal_handlers(terminate_flag)

def load_processed_files():
    """Load the list of processed files from a JSON file."""
    if os.path.exists(PROCESSED_FILES_JSON):
        try:
            with open(PROCESSED_FILES_JSON, 'r') as f:
                processed_files = set(json.load(f))
                return processed_files
        except Exception as e:
            logger.error(f"Error loading processed files: {e}")
            return set()
    else:
        return set()

def save_processed_files(processed_files):
    """Save the list of processed files to a JSON file."""
    try:
        os.makedirs(os.path.dirname(PROCESSED_FILES_JSON), exist_ok=True)
        with open(PROCESSED_FILES_JSON, 'w') as f:
            json.dump(list(processed_files), f)
    except Exception as e:
        logger.error(f"Error saving processed files: {e}")

def load_new_transcriptions():
    """Load transcription files that haven't been processed yet."""
    new_transcriptions = []
    try:
        files = sorted(os.listdir(TRANSCRIPTIONS_DIR))
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(TRANSCRIPTIONS_DIR, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    transcription = data['transcription']
                    new_transcriptions.append((timestamp, transcription, filename))
        return new_transcriptions
    except Exception as e:
        logger.error(f"Error loading transcriptions: {e}")
        return []

def should_respond(transcription):
    """Determine if the assistant should respond and which persona should respond."""
    global active_persona, last_interaction_time
    transcription_lower = transcription.lower()

    # Check if the transcription is from the assistant itself
    for response in assistant_responses:
        if transcription.strip() == response.strip():
            logger.info("Ignoring transcription as it matches an assistant's previous response.")
            return None

    # Check for activation phrases
    for persona_name, persona_data in personas.items():
        for phrase in persona_data['activation_phrases']:
            if phrase.lower() in transcription_lower:
                active_persona = persona_name
                last_interaction_time = datetime.now()
                logger.info(f"Activation phrase detected. Active persona set to '{active_persona}'.")
                return active_persona
    # Check if there's an active persona and if the conversation hasn't timed out
    if active_persona:
        if datetime.now() - last_interaction_time <= CONVERSATION_TIMEOUT:
            last_interaction_time = datetime.now()
            return active_persona
        else:
            logger.info("Conversation timed out. No active persona.")
            active_persona = None
    # If no active persona, handle default activation
    if not active_persona:
        if len(personas) == 1:
            # Only one persona is loaded, activate it
            active_persona = next(iter(personas))
            last_interaction_time = datetime.now()
            logger.info(f"No activation phrase detected. Defaulting to the only loaded persona '{active_persona}'.")
            return active_persona
        elif len(personas) > 1:
            # Multiple personas are loaded, choose one at random
            active_persona = random.choice(list(personas.keys()))
            last_interaction_time = datetime.now()
            logger.info(f"No activation phrase detected. Randomly selected persona '{active_persona}'.")
            return active_persona
    return None

def update_conversation_history(timestamp, transcription):
    """Update the conversation history, expiring old messages."""
    global conversation_history
    # Add the new message
    conversation_history.append({'timestamp': timestamp, 'role': 'user', 'content': transcription})
    # Remove messages older than CONTEXT_EXPIRATION
    now = datetime.now(tz=timestamp.tzinfo)
    cutoff_time = now - CONTEXT_EXPIRATION
    conversation_history = [msg for msg in conversation_history if msg['timestamp'] >= cutoff_time]
    # Limit the conversation history to the most recent N messages
    if len(conversation_history) > CONVERSATION_HISTORY_LIMIT:
        conversation_history = conversation_history[-CONVERSATION_HISTORY_LIMIT:]

def get_military_time():
    return datetime.now().strftime('%H:%M')

def load_persona(persona_name):
    """Load the persona data from the personas directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    personas_dir = os.path.join(project_root, 'personas')
    persona_file = os.path.join(personas_dir, f"{persona_name}.json")

    if not os.path.exists(persona_file):
        logger.error(f"Persona file '{persona_file}' does not exist.")
        sys.exit(1)

    try:
        with open(persona_file, 'r') as f:
            persona_data = json.load(f)
        prompt = persona_data.get('prompt', '')
        voices = persona_data.get('voices', {})
        activation_phrases = persona_data.get('activation_phrases', [])

        if not activation_phrases:
            logger.warning(f"No activation phrases found for persona '{persona_name}'.")
        # Check for duplicate activation phrases
        for phrase in activation_phrases:
            if phrase.lower() in activation_phrases_set:
                logger.error(f"Duplicate activation phrase '{phrase}' found in persona '{persona_name}'.")
                sys.exit(1)
            activation_phrases_set.add(phrase.lower())
        return {'prompt': prompt, 'voices': voices, 'activation_phrases': activation_phrases}
    except Exception as e:
        logger.error(f"Error loading persona '{persona_name}': {e}")
        sys.exit(1)

def load_all_personas():
    """Load all personas from the personas directory."""
    # Navigate up two directories to reach the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    personas_dir = os.path.join(project_root, 'personas')
    persona_files = [f for f in os.listdir(personas_dir) if f.endswith('.json')]

    for persona_file in persona_files:
        persona_name = os.path.splitext(persona_file)[0]
        persona_data = load_persona(persona_name)
        personas[persona_name] = persona_data

def generate_response():
    """Generate responses to transcriptions that address any loaded persona."""
    global assistant_responses  # Declare as global
    processed_files = load_processed_files()
    while not terminate_flag.is_set():
        new_transcriptions = load_new_transcriptions()

        for timestamp, transcription, filename in new_transcriptions:
            filepath = os.path.join(TRANSCRIPTIONS_DIR, filename)
            logger.info(f"New transcription: {transcription[:60]} ...")

            responding_persona = should_respond(transcription)
            if responding_persona:
                logger.info(f"Transcription will be handled by persona '{responding_persona}'.")
                persona_data = personas[responding_persona]
                # Get the provider-specific voice ID
                voice = persona_data.get('voices', {}).get(TTS_PROVIDER, DEFAULT_VOICE.get(TTS_PROVIDER))

                # Update conversation history
                update_conversation_history(timestamp, transcription)
                # Prepare messages for ChatGPT API
                military_time = get_military_time()
                prompt = persona_data['prompt'].format(military_time=military_time)

                # Use the prompt in the messages list
                messages = [
                    {
                        'role': 'system',
                        'content': prompt
                    }
                ]

                # Add conversation history
                for msg in conversation_history:
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

                # Generate response using ChatGPT API
                try:
                    completion = client.chat.completions.create(
                        model='gpt-3.5-turbo',
                        messages=messages
                    )
                    response_text = completion.choices[0].message.content.strip()
                    logger.info(f"Generated response: {response_text[:60]} ...")
                    transcription_logger.info(f"{responding_persona} | {response_text}")
                    # Add assistant's response to conversation history
                    conversation_history.append({
                        'timestamp': datetime.now(tz=timestamp.tzinfo),
                        'role': 'assistant',
                        'content': response_text
                    })
                    # Store the assistant's response to prevent self-response
                    assistant_responses.append(response_text)
                    # Limit the assistant_responses list to recent items
                    if len(assistant_responses) > 10:
                        assistant_responses = assistant_responses[-10:]

                    # Enqueue the response for transmission, include voice
                    response_queue.put({'text': response_text, 'voice': voice})

                except Exception as e:
                    logger.error(f"Error generating response: {e}")
            else:
                logger.info("No active persona to respond or message ignored. Ignoring transcription.")
            try:
                processed_filepath = os.path.join(PROCESSED_TRANSCRIPTIONS_DIR, filename)
                shutil.move(filepath, processed_filepath)
                logger.info(f"Moved processed file to {processed_filepath}")
            except Exception as e:
                logger.error(f"Error moving file {filename}: {e}")

        # Sleep briefly before checking for new transcriptions
        time.sleep(1)

def text_to_speech(text, voice):
    """Convert text to speech audio data using the configured TTS provider."""
    try:
        temp_audio_file = 'temp_response.wav'

        # Map the abstract voice name to provider-specific voice ID
        provider_voice = voice  # 'voice' is already the provider-specific voice ID

        if TTS_PROVIDER == 'openai':
            # Use OpenAI TTS API
            response = client.audio.speech.create(
                model="tts-1",
                voice=provider_voice,
                input=text,
                response_format="wav"  # Request WAV format
            )
            response.stream_to_file(temp_audio_file)

        elif TTS_PROVIDER == 'unrealspeech':
            # Use UnrealSpeech TTS API
            if not UNREALSPEECH_API_KEY:
                logger.error("UnrealSpeech API key is not set. Please set it in config/settings.py.")
                return None

            url = 'https://api.v7.unrealspeech.com/stream'
            headers = {
                'Authorization': f'Bearer {UNREALSPEECH_API_KEY}'
            }
            data = {
                'Text': text,
                'VoiceId': provider_voice,
                'Bitrate': '192k',
                'Speed': '0',
                'Pitch': '1',
                'Codec': 'libmp3lame'  # Receive MP3 format
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                temp_mp3_file = 'temp_response.mp3'
                # Save the content to temp_mp3_file
                with open(temp_mp3_file, 'wb') as f:
                    f.write(response.content)
                # Convert MP3 to WAV using ffmpeg
                command = ['ffmpeg', '-y', '-i', temp_mp3_file, temp_audio_file]
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Remove the temp MP3 file
                os.remove(temp_mp3_file)
            else:
                logger.error(f"UnrealSpeech API request failed with status code {response.status_code}: {response.text}")
                return None
        else:
            logger.error(f"Unknown TTS provider: {TTS_PROVIDER}")
            return None

        if debug_mode:
            os.makedirs(TTS_AUDIO_DIR, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            debug_audio_file = os.path.join(TTS_AUDIO_DIR, f"response_{timestamp}.wav")
            os.rename(temp_audio_file, debug_audio_file)
            logger.info(f"Saved TTS audio to {debug_audio_file}")
            return debug_audio_file

        return temp_audio_file
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}")
        return None

def play_audio(audio_file):
    """Play audio file using the appropriate player."""
    try:
        logger.info(f"Playing audio file: {audio_file} on device: {AUDIO_DEVICE}")
        create_lock()  # Signal the receiver to pause

        # Determine the file extension
        _, ext = os.path.splitext(audio_file)
        ext = ext.lower()

        if ext == '.wav':
            player = ['aplay', '-D', AUDIO_DEVICE, audio_file]
        elif ext == '.mp3':
            player = ['mpg123', '-a', AUDIO_DEVICE, audio_file]
        else:
            logger.error(f"Unsupported audio format: {ext}")
            return

        subprocess.run(player, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Audio playback completed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error playing audio: {e}")
    finally:
        time.sleep(2)
        remove_lock()  # Signal the receiver to resume
        if not debug_mode and os.path.exists(audio_file):
            os.remove(audio_file)
            logger.info(f"Removed temporary audio file: {audio_file}")

def transmit_responses():
    """Transmit responses from the response queue."""
    while not terminate_flag.is_set():
        try:
            response_item = response_queue.get(timeout=1)
            if response_item:
                response_text = response_item['text']
                voice = response_item['voice']
                logger.debug(f"Transmitting response: {response_text}")
                audio_file = text_to_speech(response_text, voice)
                if audio_file:
                    play_audio(audio_file)
                else:
                    logger.error("Failed to convert text to speech.")
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in transmit_responses: {e}")

def main():
    """Main function to run the transmitter."""
    global debug_mode

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Dispatcher AI transmitter script.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to save TTS audio files.')
    parser.add_argument('--persona', type=str, action='append', help='Specify the persona(s) to use.')
    parser.add_argument('--load-all-personas', action='store_true', help='Load all available personas.')
    args = parser.parse_args()

    debug_mode = args.debug

    if debug_mode:
        logger.info("Debug mode is enabled. TTS audio responses will be saved.")

    # Load personas
    if args.load_all_personas:
        logger.info("Loading all available personas.")
        load_all_personas()
    elif args.persona:
        for persona_name in args.persona:
            logger.info(f"Loading persona '{persona_name}'.")
            persona_data = load_persona(persona_name)
            personas[persona_name] = persona_data
    else:
        logger.error("No personas specified. Use '--persona' or '--load-all-personas'.")
        sys.exit(1)

    if not personas:
        logger.error("No personas loaded. Exiting.")
        sys.exit(1)

    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        logger.info("Removed stale lock file on startup.")

    # Print startup message
    loaded_personas = ', '.join(personas.keys())
    logger.info(f"Dispatch Alpha transmitter script started with personas: {loaded_personas}. Ready to process transcriptions.")

    # Start threads for generating responses and transmitting
    generator_thread = threading.Thread(target=generate_response)
    transmitter_thread = threading.Thread(target=transmit_responses)
    generator_thread.daemon = True
    transmitter_thread.daemon = True
    generator_thread.start()
    transmitter_thread.start()

    try:
        while not terminate_flag.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        terminate_flag.set()
    finally:
        generator_thread.join()
        transmitter_thread.join()
        logger.info("Cleanup complete. Exiting.")

if __name__ == '__main__':
    main()
