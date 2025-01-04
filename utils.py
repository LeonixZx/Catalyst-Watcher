import os
import json

def load_api_settings():
    if not os.path.exists('api_settings.json'):
        default_settings = {
            "AI Community Platform": {"platform": "AI Community Platform", "token": ""},
            "OpenAI ChatGPT": {"platform": "OpenAI ChatGPT", "token": ""}
        }
        with open('api_settings.json', 'w') as f:
            json.dump(default_settings, f, indent=2)
        return default_settings
    else:
        with open('api_settings.json', 'r') as f:
            return json.load(f)