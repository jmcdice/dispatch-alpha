# src/transmitter/tx_stable.py

import os
import sys
import shutil
import json
import time
import logging
import threading
from datetime import datetime, timedelta
import queue
import argparse
import warnings

import subprocess

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
    VOICE_NAME,
    TTS_AUDIO_DIR,
    CONTEXT_EXPIRATION,
    RESPONSE_QUEUE_MAX_SIZE,
    CONVERSATION_HISTORY_LIMIT,
    TX_LOG_FILE,
    LOG_FORMAT,
    TRANSCRIPTIONS_DIR,
    PROCESSED_TRANSCRIPTIONS_DIR
)
from src.common.utils import initialize_logging, register_signal_handlers

from openai import OpenAI

# Initialize Logging
initialize_logging(TX_LOG_FILE, LOG_FORMAT)
logger = logging.getLogger(__name__)

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
    """Check if the transcription addresses the dispatcher."""
    transcription_lower = transcription.lower()
    if 'dispatch alpha' in transcription_lower or 'dispatch, alpha' in transcription_lower:
        return True
    return False

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

def generate_response():
    """Generate responses to transcriptions that address the dispatcher."""
    processed_files = load_processed_files()
    while not terminate_flag.is_set():
        new_transcriptions = load_new_transcriptions()

        for timestamp, transcription, filename in new_transcriptions:
            filepath = os.path.join(TRANSCRIPTIONS_DIR, filename)
            logger.info(f"New transcription: {transcription[:60]} ...")

            if should_respond(transcription):
                logger.info("Transcription addresses the dispatcher. Generating response.")
                # Update conversation history
                update_conversation_history(timestamp, transcription)
                # Prepare messages for ChatGPT API
                military_time = get_military_time()
                prompt = f"""
You are Dispatch Alpha, a professional radio dispatcher. But never say Alpha, only Dispatch.
The current military time is {military_time}.
Your job is to communicate with other radio operators using formal and proper radio etiquette.
Ensure all responses are concise, clear, and use common radio phrases such as "Copy" and "Over."
You must always close your transmission with the current military time by saying,
"Dispatch out, {military_time}."
"""

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
                    # Add assistant's response to conversation history
                    conversation_history.append({'timestamp': datetime.now(tz=timestamp.tzinfo), 'role': 'assistant', 'content': response_text})
                    # Enqueue the response for transmission
                    response_queue.put(response_text)
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
            else:
                logger.info("Transcription does not address the dispatcher. Ignoring.")
                # Optionally, still update conversation history without assistant response
                update_conversation_history(timestamp, transcription)
            try:
                processed_filepath = os.path.join(PROCESSED_TRANSCRIPTIONS_DIR, filename)
                shutil.move(filepath, processed_filepath)
                logger.info(f"Moved processed file to {processed_filepath}")
            except Exception as e:
                logger.error(f"Error moving file {filename}: {e}")

        # Sleep briefly before checking for new transcriptions
        time.sleep(1)

def text_to_speech(text):
    """Convert text to speech audio data using OpenAI's TTS model, requesting WAV format."""
    try:
        temp_audio_file = 'temp_response.wav'

        response = client.audio.speech.create(
            model="tts-1",
            voice=VOICE_NAME,
            input=text,
            response_format="wav"  # Request WAV format
        )
        response.stream_to_file(temp_audio_file)

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
    """Play audio file using aplay with the specified device."""
    try:
        logger.info(f"Playing audio file: {audio_file} on device: {AUDIO_DEVICE}")
        #subprocess.run(['aplay', '-D', AUDIO_DEVICE, audio_file], check=True)
        subprocess.run(['aplay', '-D', AUDIO_DEVICE, audio_file], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logger.info("Audio playback completed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error playing audio: {e}")
    finally:
        if not debug_mode and os.path.exists(audio_file):
            os.remove(audio_file)
            logger.info(f"Removed temporary audio file: {audio_file}")

def transmit_responses():
    """Transmit responses from the response queue."""
    while not terminate_flag.is_set():
        try:
            response_text = response_queue.get(timeout=1)
            if response_text:
                logger.debug(f"Transmitting response: {response_text}")
                audio_file = text_to_speech(response_text)
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
    args = parser.parse_args()

    debug_mode = args.debug

    if debug_mode:
        logger.info("Debug mode is enabled. TTS audio responses will be saved.")

    # Print startup message
    logger.info("Dispatcher AI transmitter script started in VOX mode. Ready to process transcriptions.")

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

