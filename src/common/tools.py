import re
import os
import requests
from config.settings import WEATHER_API_KEY

class Tool:
    def __init__(self, name, trigger_patterns, action_function):
        self.name = name
        self.trigger_patterns = trigger_patterns  # List of regex patterns
        self.action_function = action_function

    def check_trigger(self, text):
        for pattern in self.trigger_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def execute(self, text):
        return self.action_function(text)

# Define the action function for the weather tool
def get_weather(text):
    # Extract location if mentioned
    match = re.search(r'weather in ([\w\s]+)', text, re.IGNORECASE)
    if match:
        location = match.group(1).strip()
    else:
        location = 'Boulder, colorado'  # Default location or handle appropriately

    # Call the weather API
    if not WEATHER_API_KEY:
        return "Weather API key is not set."

    api_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(api_url)
        data = response.json()

        if data.get('cod') != 200:
            return f"Could not retrieve weather data for {location}."

        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"The current weather in {location} is {weather_description} with a temperature of {temperature}°C."
    except Exception as e:
        return f"Error fetching weather data: {e}"

# Initialize the tools registry
tools_registry = [
    Tool(
        name='Weather',
        trigger_patterns=[
            r"what's the weather",
            r"what is the weather",
            r"weather today",
            r"weather in [\w\s]+",
            r"current weather"
        ],
        action_function=get_weather
    ),
    # Add more tools here
]

