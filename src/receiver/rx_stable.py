# src/receiver/rx_stable.py

import argparse
import io
import json
import logging
import os
import queue
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from src.common.tools import tools_registry

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI  # Updated import
from logging.handlers import WatchedFileHandler


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import settings and utilities
from config.settings import (
    API_KEY,
    SAMPLE_RATE,
    CHANNELS,
    AUDIO_DEVICE_INDEX,
    AUDIO_THRESHOLD,
    SILENCE_DURATION_THRESHOLD,
    MIN_RECORDING_DURATION,
    MAX_RECORDING_DURATION,
    PRE_ROLL_DURATION,
    POST_ROLL_DURATION,
    QUEUE_MAX_SIZE,
    TRANSCRIPTION_WORKERS,
    TRANSCRIPTIONS_DIR,
    AUDIO_DIR,
    LOG_FILE,
    LOG_FORMAT,
    TRANSCRIPTIONS_LOG_FILE
)

from src.common.utils import list_audio_devices, initialize_logging

# Initialize Logging
initialize_logging(LOG_FILE, LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize Transcription Logger
transcription_logger = logging.getLogger('transcriptions')
transcription_logger.setLevel(logging.INFO)
transcription_handler = WatchedFileHandler(TRANSCRIPTIONS_LOG_FILE)
transcription_handler.setLevel(logging.INFO)
transcription_formatter = logging.Formatter('%(asctime)s | %(processName)s | user | %(message)s')
transcription_handler.setFormatter(transcription_formatter)
transcription_logger.addHandler(transcription_handler)

# Initialize OpenAI Client
if not API_KEY:
    logger.error("OPENAI_API_KEY is not set. Please set it as an environment variable.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)  # Instantiate the OpenAI client

# Global Variables
audio_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
terminate_flag = threading.Event()

# Lock file path for coordination with transmitter
LOCK_FILE = '/tmp/tx_rx_lock'

def test_audio_input():
    """Test audio input by recording for a short duration and measuring RMS."""
    duration = 2  # seconds
    logger.info(f"Testing audio input. Recording for {duration} seconds...")
    try:
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            device=AUDIO_DEVICE_INDEX,
            dtype='float32'
        )
        sd.wait()
        rms = np.sqrt(np.mean(np.square(recording)))
        logger.info(f"Test recording RMS: {rms}")
        if rms < 0.001:
            logger.warning("Very low audio levels detected. Check microphone and audio settings.")
        else:
            logger.info("Audio input test completed successfully.")
    except Exception as e:
        logger.error(f"Error during audio input test: {e}")
        sys.exit(1)

def signal_handler(sig, frame):
    """Handle termination signals to allow graceful shutdown."""
    logger.info("Termination signal received. Cleaning up...")
    terminate_flag.set()

# Register Signal Handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def transcribe_audio(audio_data):
    """Transcribe audio data using OpenAI's Whisper API and process tools."""
    try:
        with io.BytesIO() as audio_buffer:
            # Write the numpy array to the buffer in WAV format with PCM_16 subtype
            sf.write(audio_buffer, audio_data, SAMPLE_RATE, format='WAV', subtype='PCM_16')
            audio_buffer.seek(0)
            audio_buffer.name = 'audio.wav'  # Set the name attribute for format recognition

            # Transcribe using OpenAI API
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_buffer
            )
            transcription_text = transcription.text.strip()

            # Skip logging and saving if transcription is empty
            if not transcription_text or transcription_text.lower() == 'you':
                logger.info(f"Skipped transcription: '{transcription_text}' (empty or irrelevant).")
                return

            # Log the transcription
            transcription_logger.info(f"{transcription_text}")

            # Process the transcription to check for tool triggers
            tool_response = process_transcription_text(transcription_text)
            if tool_response:
                # Log the tool response
                transcription_logger.info(f"Tool Response: {tool_response}")
                logger.info(f"Tool executed. Response: {tool_response}")

            # Save the transcription to a JSON file
            os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
            timestamp = datetime.now().isoformat()
            transcription_data = {
                'timestamp': timestamp,
                'transcription': transcription_text
            }
            filename = f"transcription_{timestamp.replace(':', '-')}.json"
            filepath = os.path.join(TRANSCRIPTIONS_DIR, filename)
            with open(filepath, 'w') as json_file:
                json.dump(transcription_data, json_file)
            logger.info(f"Saved transcription to {filepath}")

    except Exception as e:
        logger.error(f"Error during transcription: {e}")

def process_transcription_text(transcription_text):
    """Process transcription text to check for tool triggers."""
    for tool in tools_registry:
        if tool.check_trigger(transcription_text):
            logger.info(f"Trigger found for tool: {tool.name}")
            response = tool.execute(transcription_text)
            return response
    return None


def process_audio(executor, debug_mode):
    """Process audio data from the queue and handle transcription."""
    while not terminate_flag.is_set():
        try:
            audio_data = audio_queue.get(timeout=1)
            if audio_data is None:
                continue  # Skip if None

            # Save audio to disk if in debug mode
            if debug_mode:
                save_audio_to_disk(audio_data)

            # Submit transcription task to the executor
            executor.submit(transcribe_audio, audio_data)

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in processing audio: {e}")

def save_audio_to_disk(audio_data):
    """Save the provided audio data to the audio directory with a timestamp."""
    try:
        os.makedirs(AUDIO_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"audio_{timestamp}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        sf.write(filepath, audio_data, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        logger.info(f"Saved audio file to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")

def audio_callback(indata, frames, time_info, status):
    """Callback function called by the audio stream for each audio block."""
    if status:
        logger.warning(f"Audio Stream Status: {status}")

    # Check if the transmitter is transmitting (lock file exists)
    if os.path.exists(LOCK_FILE):
        # If we are currently recording, we need to reset the state
        if hasattr(audio_callback, "initialized") and audio_callback.initialized:
            if audio_callback.recording:
                logger.info("Transmitter is active. Resetting recording state.")
                audio_callback.recording = False
                audio_callback.audio_frames = []
                audio_callback.pre_roll_buffer = []
                audio_callback.post_roll_counter = 0.0
                audio_callback.silence_counter = 0.0
                audio_callback.recording_duration = 0.0
        # Skip processing
        return

    rms = np.sqrt(np.mean(np.square(indata)))

    # Initialize stateful variables on first call
    if not hasattr(audio_callback, "initialized"):
        audio_callback.initialized = True
        audio_callback.recording = False
        audio_callback.audio_frames = []
        audio_callback.pre_roll_buffer = []
        audio_callback.post_roll_counter = 0.0
        audio_callback.silence_counter = 0.0
        audio_callback.recording_duration = 0.0

    # Update pre-roll buffer
    audio_callback.pre_roll_buffer.append(indata.copy())
    pre_roll_max_frames = int(PRE_ROLL_DURATION * SAMPLE_RATE / frames)
    if len(audio_callback.pre_roll_buffer) > pre_roll_max_frames:
        audio_callback.pre_roll_buffer.pop(0)

    if not audio_callback.recording and rms > AUDIO_THRESHOLD:
        audio_callback.recording = True
        logger.info("Sound detected, start recording")
        audio_callback.audio_frames = audio_callback.pre_roll_buffer.copy()
        audio_callback.post_roll_counter = 0.0
        audio_callback.silence_counter = 0.0
        audio_callback.recording_duration = 0.0

    if audio_callback.recording:
        audio_callback.audio_frames.append(indata.copy())
        audio_callback.recording_duration += frames / SAMPLE_RATE

        if rms > AUDIO_THRESHOLD:
            audio_callback.silence_counter = 0.0
            audio_callback.post_roll_counter = 0.0
        else:
            audio_callback.silence_counter += frames / SAMPLE_RATE
            audio_callback.post_roll_counter += frames / SAMPLE_RATE

        # Check if we should stop recording
        if (audio_callback.silence_counter >= SILENCE_DURATION_THRESHOLD and
                audio_callback.post_roll_counter >= POST_ROLL_DURATION and
                audio_callback.recording_duration >= MIN_RECORDING_DURATION):
            audio_callback.recording = False
            logger.info("Silence detected and post-roll duration met, stop recording")
            audio_data = np.concatenate(audio_callback.audio_frames)
            try:
                audio_queue.put_nowait(audio_data)
                logger.info("Queued audio data for transcription")
            except queue.Full:
                logger.warning("Audio queue is full. Dropping audio data.")
            # Clear frames and pre-roll buffer
            audio_callback.audio_frames = []
            audio_callback.pre_roll_buffer = []
        elif audio_callback.recording_duration >= MAX_RECORDING_DURATION:
            audio_callback.recording = False
            logger.info("Maximum recording duration reached, stop recording")
            audio_data = np.concatenate(audio_callback.audio_frames)
            try:
                audio_queue.put_nowait(audio_data)
                logger.info("Queued audio data for transcription")
            except queue.Full:
                logger.warning("Audio queue is full. Dropping audio data.")
            # Clear frames and pre-roll buffer
            audio_callback.audio_frames = []
            audio_callback.pre_roll_buffer = []

def main():
    """Main function to run the audio recording and transcription."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Audio recording and transcription script.')
    parser.add_argument('--test', action='store_true', help='Run in test mode, record for 5 seconds and exit.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to save audio files.')
    args = parser.parse_args()

    debug_mode = args.debug

    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        logger.info("Removed stale lock file on startup.")

    # Test audio input before starting
    test_audio_input()

    if args.test:
        logger.info("Test mode active. Exiting after test.")
        sys.exit(0)

    # Initialize ThreadPoolExecutor for transcription
    with ThreadPoolExecutor(max_workers=TRANSCRIPTION_WORKERS) as executor:
        # Start audio processing thread
        processing_thread = threading.Thread(target=process_audio, args=(executor, debug_mode))
        processing_thread.daemon = True
        processing_thread.start()

        try:
            with sd.InputStream(
                device=AUDIO_DEVICE_INDEX,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                callback=audio_callback
            ):
                logger.info("Audio stream started. Listening for incoming transmissions...")

                while not terminate_flag.is_set():
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Error in audio stream: {e}")
        finally:
            terminate_flag.set()
            processing_thread.join()
            executor.shutdown(wait=True)
            logger.info("Cleanup complete. Exiting.")

if __name__ == '__main__':
    list_audio_devices()
    main()

