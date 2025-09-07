import requests
from utils import wolfram_app_id

def get_daily_weather_summary():
    """
    Gets a summary of today's weather in Palo Alto, CA using Wolfram|Alpha.
    """
    if not wolfram_app_id or wolfram_app_id == "YOUR_APP_ID":
        return "Weather information is unavailable."
    
   
    api_url = f"http://api.wolframalpha.com/v1/result?appid={wolfram_app_id}&i=weather%20in%20Palo%20Alto%20CA"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        weather = response.text
   
        return f"The weather forecast for today is: {weather}."
    except requests.exceptions.RequestException as e:
        print(f"Error querying Wolfram|Alpha for weather: {e}")
        return "The weather forecast for today is unavailable."