# src/common/utils.py

import logging
import signal

def list_audio_devices():
    """List available audio input devices."""
    logging.info("Listing available audio devices:")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            io_type = "Input" if device['max_input_channels'] > 0 else "Output"
            logging.info(f"Device {idx}: {device['name']} - {device['hostapi']} - {io_type}")
    except Exception as e:
        logging.error(f"Failed to list audio devices: {e}")

def initialize_logging(log_file, log_format):
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def register_signal_handlers(terminate_flag):
    """Register signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logging.info("Termination signal received. Cleaning up...")
        terminate_flag.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

