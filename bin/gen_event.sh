##!/bin/bash

# Directory where JSON files will be saved
# Adjusted the path to point to the new 'data/transcriptions' directory
TRANSCRIPTIONS_DIR="$(dirname "$0")/../data/transcriptions"

# Create the directory if it doesn't exist
mkdir -p "$TRANSCRIPTIONS_DIR"

# Function to get the current timestamp for the JSON content
get_timestamp_json() {
  date '+%Y-%m-%dT%H:%M:%S'
}

# Function to get a timestamp for the filename (with milliseconds)
get_timestamp_filename() {
  date '+%Y-%m-%dT%H-%M-%S-%3N'
}

# List of dispatcher-related quotes with 10 additional events
QUOTES=(
  "Dispatch Alpha, this is Station 3. We are responding to a fire at 1427 Maple Avenue. Heavy smoke visible. Send additional units. Over."
  "Dispatch Alpha, this is Unit 12. Code 4 on the patrol route. All clear at the assigned checkpoints. Over."
  "Dispatch Alpha, this is Unit 24. We've got a traffic stop on Highway 36. License plate is coming back expired. Please advise. Over."
  "Dispatch Alpha, this is Unit 52. I'm out with a suspicious vehicle on 8th and Redwood. Requesting backup. Over."
  "Dispatch Alpha, this is Base Command. Weather advisory issued for the northern region. Expect heavy rain and possible flooding. Alert all units. Over."
  "Dispatch Alpha, this is Unit 9. We've got a report of a missing child, last seen near the park. Requesting search teams. Over."
  "Dispatch Alpha, this is Unit 15. We're setting up a roadblock at the intersection of 1st and Cedar. Request additional patrol cars. Over."
  "Dispatch Alpha, this is Unit 27. We have an accident involving three vehicles on Main Street. Medical assistance needed. Over."
  "Dispatch Alpha, this is Unit 34. Surveillance indicates possible break-in at the warehouse district. Initiating investigation. Over."
  "Dispatch Alpha, this is Unit 11. Traffic lights are out at 5th and Pine due to power outage. Requesting maintenance crew. Over."
  "Dispatch Alpha, this is Rescue Team 5. We've reached the avalanche site. Beginning search and rescue operations. Over."
  "Dispatch Alpha, this is Unit 20. Pursuing a suspect heading east on Highway 10. Requesting air support. Over."
  "Dispatch Alpha, this is Fire Squad 8. Fire contained at the industrial park. Performing safety checks. Over."
  "Dispatch Alpha, this is Unit 14. Found unattended package at the train station. Bomb squad needed. Over."
  "Dispatch Alpha, this is Unit 3. We are escorting a VIP convoy to the city hall. Monitoring surroundings. Over."
  "Dispatch Alpha, this is Coast Guard Unit 7. We have a distress signal from a vessel 5 miles offshore. Deploying rescue boats. Over."
  "Dispatch Alpha, this is Unit 19. Gas leak detected in the residential area. Evacuating civilians. Over."
  "Dispatch Alpha, this is Air Patrol 2. Severe turbulence reported in our sector. Adjusting flight path. Over."
  "Dispatch Alpha, this is Unit 6. Lost visual on the suspect. Requesting additional units to establish perimeter. Over."
  "Dispatch Alpha, this is Medical Team 2. En route to the scene with estimated arrival in 5 minutes. Over."
)

# Pick a random quote
RANDOM_QUOTE="${QUOTES[$RANDOM % ${#QUOTES[@]}]}"

# Get the current timestamp for the JSON content
TIMESTAMP_JSON=$(get_timestamp_json)

# Get the timestamp for the filename
TIMESTAMP_FILENAME=$(get_timestamp_filename)

# Generate filename with a different timestamp format (including milliseconds)
FILENAME="$TRANSCRIPTIONS_DIR/transcription_$TIMESTAMP_FILENAME.json"

# Create the JSON content
JSON_CONTENT=$(cat <<EOF
{
  "timestamp": "$TIMESTAMP_JSON",
  "transcription": "$RANDOM_QUOTE"
}
EOF
)

# Write the JSON content to the file
echo "$JSON_CONTENT" > "$FILENAME"

echo "Generated: $FILENAME"
