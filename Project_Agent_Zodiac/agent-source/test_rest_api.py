import requests
import google.auth
from google.auth.transport.requests import Request
import json
import logging

logging.basicConfig(level=logging.DEBUG)

def test_agent():
    print("Getting credentials...")
    credentials, project = google.auth.default()
    credentials.refresh(Request())
    
    token = credentials.token
    
    agent_id = "projects/595396735241/locations/us-central1/reasoningEngines/4000882020430381056"
    url = f"https://us-central1-aiplatform.googleapis.com/v1beta1/{agent_id}:list_sessions"
    
    print(f"Querying URL: {url}")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Try different payload structures just in case
    payload = {}
    
    print("Payload:", json.dumps(payload, indent=2))
    
    response = requests.post(url, headers=headers, json=payload)
    
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)

if __name__ == "__main__":
    test_agent()
