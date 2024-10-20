#!/usr/bin/env bash

set -e

log() {
  echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" 
}

# Function to mute the microphone
mute_mic() {
  amixer -c 2 set Mic nocap > /dev/null
  log "Microphone muted."
}

# Function to unmute the microphone
unmute_mic() {
  amixer -c 2 set Mic cap > /dev/null
  log "Microphone unmuted."
}

# Function to play the specified audio file with controlled transmission
play_audio() {
  local AUDIO_FILE=$1
  # mute_mic
  log "Starting audio playback..."
  if aplay -D plughw:1,0 "$AUDIO_FILE"; then
    log "Playback successful."
  else
    log "Playback failed."
    # unmute_mic
    exit 1
  fi
}

# Function to display usage instructions
usage() {
  echo "Usage: $0 <audio_file>"
  echo ""
  echo "Arguments:"
  echo "  audio_file    Path to the audio file to be played."
  echo ""
  echo "Options:"
  echo "  -h, --help    Display this help message."
}

# Parse command-line arguments
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
  exit 0
fi

AUDIO_FILE=$1

# Check if the audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
  log "Audio file '$AUDIO_FILE' does not exist."
  exit 1
fi

# Main Execution
log "Playing audio file: $AUDIO_FILE"
play_audio "$AUDIO_FILE"
log "Operation completed successfully."
