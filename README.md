# Dispatch Alpha

Dispatch Alpha is an intelligent radio dispatch system that integrates Baofeng walkie-talkies with a Raspberry Pi and AI capabilities. Using OpenAI's Whisper API for transcription, ChatGPT for generating responses, and Text-to-Speech (TTS0-1) for audio output, the system can transcribe radio communications, analyze them, and respond to user inputs on the same radio channel.

[Watch the demo](https://storage.googleapis.com/echo-004/dispatch-alpha-demo.mov)

## Overview

Dispatch Alpha aims add a dispatcher to any channel by introducing AI capabilities from OpenAI to traditional radio systems. Currently using two baofengs and a raspberry pi. You also need an additional baofeng for just listeing in. I'm using these and they're great: 

[Baofeng Radios](https://www.amazon.com/dp/B0CW2QQGXD) $70 for two radios

[USB Sound Card](https://www.amazon.com/dp/B00IRVQ0F8) $7

[2.5mm to 3.5mm audio cable](https://www.amazon.com/Mr-Rex-Replacement-Compatible-Cancelling/dp/B07W6FD6MV/) $6

- **Receiver (RX)**: Listens to the radio channel, captures incoming audio, and transcribes it using OpenAI's Whisper API.
- **Transmitter (TX)**: Generates AI-based responses using OpenAI's ChatGPT and TTS APIs, and transmits them back over the radio channel.

This is what my setup looks like:

```console
    │
[Raspberry Pi]
    │
    - [USB Sound Card] -> [RX Baofeng (Receiver)]
    │ - 2.5mm audio cable (Speaker Output) -> Mic-In (Red Port)
    │ (RX code listens here src/reciever/rx-stable.py) 
    │
    - [ BCM2835 Headphone Jack (speaker output) ] -> [TX Baofeng (Transmitter)]
    │ - 3.5mm audio cable -> Mic Input of Baofeng
    │ (TX code listens here src/transmitter/tx-stable.py)
    │
```

## Features

- Real-Time Transcription: Captures and transcribes incoming radio communications.
- AI-Powered Responses: Generates context-aware responses using OpenAI's ChatGPT.
- Text-to-Speech Conversion: Converts AI-generated text responses to speech for radio transmission.
- Conversation History: Maintains a conversation history to provide context-aware interactions.
- Configurable Settings: Adjustable thresholds and parameters for audio processing.
- Logging: Comprehensive logging of transcriptions and system operations.

## Hardware Setup

### Components

- Raspberry Pi (using an rpi4 with 1G RAM and 128 SD disk)
- Two Baofeng Walkie-Talkies
- USB Sound Card
- Audio Cables and Adapters

### Connections

#### Receiver (RX) Baofeng:
- Speaker Output (2.5mm jack): Connects to the Mic-In (red port) on the USB sound card using a 2.5mm male-to-male cable or appropriate adapter.
- PTT (Push-To-Talk): Not required for RX.

#### Transmitter (TX) Baofeng:
- Mic Input (3.5mm jack): Connects to the Raspberry Pi's audio output (headphone jack or USB sound card output).
- PTT Control: Configured to use VOX mode for automatic transmission upon detecting audio output from the Raspberry Pi.

#### USB Sound Card:
- Plugs into one of the Raspberry Pi's USB ports.
- Provides mic-in and audio-out capabilities.

#### Connection Diagram
```
[RX Baofeng Speaker Output] --(2.5mm audio cable)--> [USB Sound Card Mic-In (Red Port)] --(USB)--> [Raspberry Pi]

[Raspberry Pi Audio Output] --(3.5mm audio cable)--> [TX Baofeng Mic Input (3.5mm jack)]
```

## Software Setup

### Requirements

- Raspberry Pi with Raspbian OS (or similar Debian-based OS)
- Python 3.7 or higher
- OpenAI API Access (for Whisper, ChatGPT, and TTS APIs)
- Python Libraries:
  - numpy
  - sounddevice
  - soundfile
  - openai
  - asyncio
  - threading
  - logging

### Configuration

1. **OpenAI API Key**: Obtain an API key from OpenAI and set it as an environment variable:
   ```bash
   export OPENAI_API_KEY='your_actual_api_key_here'
   ```

2. **Audio Device Index**: Adjust the `AUDIO_DEVICE_INDEX` in both `rx-stable.py` and `tx-stable.py` to match your audio devices. You can list audio devices using the `list_audio_devices()` function or by running:
   ```bash
   python3 -c "import sounddevice as sd; print(sd.query_devices())"
   ```

3. **Adjust Thresholds**: Modify the `AUDIO_THRESHOLD`, `SILENCE_DURATION_THRESHOLD`, and other constants in the scripts to fine-tune audio processing based on your environment.

### Installation

1. Clone the Repository
   ```bash
   git clone https://github.com/jmcdice/dispatch-alpha.git
   cd dispatch-alpha
   ```

2. Set Up the Environment
   Ensure your Raspberry Pi is up to date:
   ```bash
   sudo apt-get update
   sudo apt-get upgrade
   ```

3.  Install Python libraries:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip3 install -r requirements.txt
   ```

4. Configure Audio Devices
   List available audio devices:
   ```bash
   python3 -c "import sounddevice as sd; print(sd.query_devices())"
   ```
   Identify the indexes of your input and output audio devices and update config/settings.py:
   ```bash

   AUDIO_DEVICE_INDEX = 2  # This is for RX
   TX_AUDIO_DEVICE_INDEX = 1  # This is for TX
   AUDIO_DEVICE = "plughw:1,0"  # ALSA name for the onboard headphone jack
   ```

5. Set Up OpenAI API Key

   Set your OpenAI API key as an environment variable:
   ```bash
      export OPENAI_API_KEY='your_actual_api_key_here'
   ```

### Receiver (RX) Script

The `rx-stable.py` script captures audio from the RX Baofeng, sends it to OpenAI's Whisper API for transcription, and saves the transcribed text.

Running the RX Script:
```bash
bash scripts/run_receiver.sh
```

Command-Line Options:
- `--test`: Run in test mode, records for 5 seconds, and exits.
- `--debug`: Enable debug mode to save audio files to disk.

Functionality:
- Audio Capture: Captures audio from the specified input device.
- Voice Activity Detection: Detects when someone is speaking based on audio levels.
- Transcription: Sends audio chunks to OpenAI's Whisper API for transcription.
- Logging: Logs transcriptions and saves them to the `transcriptions` directory in JSON format.

### Transmitter (TX) Script

NOTES ON TX: 
 - Configure VOX mode on the Baofeng. Sensitivity is set between 0 (off) and 10. With 1 being the most sensitive. Choose 1.
 - Using `alsamixer` set the volume of the on-board headphone jack to 90+ to trigger the VOX.

The `tx-stable.py` script monitors transcriptions, generates responses using OpenAI's ChatGPT, converts the responses to speech using OpenAI's TTS API, and plays the audio back through the TX Baofeng.

Running the TX Script:
```bash
bash scripts/run_transmitter.sh
```

Command-Line Options:
- `--debug`: Enable debug mode to save TTS audio files to disk.

Functionality:
- Monitoring Transcriptions: Watches the `transcriptions` directory for new transcriptions.
- AI Response Generation: Uses OpenAI's ChatGPT to generate context-aware responses when addressed.
- Text-to-Speech Conversion: Converts AI-generated text responses to speech using OpenAI's TTS API.
- Audio Playback: Plays the audio responses through the specified output device.

Note: The script assumes that the TX Baofeng is set to VOX mode to automatically transmit when audio is played.

Example Interaction:
1. User: "Dispatch Alpha, we have a situation at the main gate."
2. System: (Transcribes the message and detects that it addresses the dispatcher.)
3. System: Generates a response: "Copy that. Dispatch out, 14:35."
4. System: Converts the response to speech and transmits it over the radio.

## Future Development

Thinking about some additions..


- Enhanced AI Capabilities: Analyze transcriptions for commands, keywords, or emergency signals.
- Web Dashboard: Develop a web-based interface for monitoring and controlling the system.
- Custom Voice Models: Integrate custom TTS voices for more personalized interactions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

MIT License
