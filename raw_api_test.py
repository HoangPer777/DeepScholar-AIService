import requests
import os
import json
from dotenv import load_dotenv

dotenv_path = r"d:\aaa\aa_study_document\a_year 4 semester 2\Graduation Project\DeepScholar\DeepScholar-AIService\.env"
load_dotenv(dotenv_path)

api_key = os.getenv("GOOGLE_API_KEY")

def test_endpoint(version, model):
    url = f"https://generativelanguage.googleapis.com/{version}/{model}:embedContent?key={api_key}"
    payload = {
        "model": model,
        "content": {
            "parts": [{"text": "Hello world"}]
        }
    }
    headers = {'Content-Type': 'application/json'}
    
    print(f"\nTesting {version} with {model}...")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            # print(response.json())
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if not api_key:
    print("No API Key found")
else:
    # Test common combinations
    test_endpoint("v1", "models/text-embedding-004")
    test_endpoint("v1beta", "models/text-embedding-004")
    test_endpoint("v1", "models/embedding-001")
    test_endpoint("v1beta", "models/embedding-001")
