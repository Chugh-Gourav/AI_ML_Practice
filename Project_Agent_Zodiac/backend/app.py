import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import sys
import google.auth
from google.auth.transport.requests import Request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Zodiac Travel Agent API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# Initialize Cloud Agent ID
PROJECT_ID = "595396735241"
LOCATION = "us-central1"
# NEW AGENT ID specified by user
AGENT_ID = "3538706705741250560"
# Using IAM Auth instead of API Key
REASONING_ENGINE_URL = f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{AGENT_ID}:query"

def get_auth_token():
    creds, _ = google.auth.default()
    creds.refresh(Request())
    return creds.token

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str = None
    history: list = [] 

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Determine strict message to send
        # We rely on the Agent's handling of 'history' now for memory.
        # But we still send CRITICAL instructions in the message to ensure adherence.
        
        final_input_message = (
            "CRITICAL INSTRUCTIONS:\n"
            "- **TONE**: Be exciting, cosmic, and engaging! Use emojis (e.g., ✨, 🌌, ✈️) to make the response feel magical.\n"
            "- **FORMAT**: Your itineraries MUST be presented as a simple list of single-line items. Do NOT use complex markdown or paragraphs for the itinerary list. \n"
            "  - Example: `• Day 1: Arrive in Paris and visit the Eiffel Tower`\n"
            "- **CONTEXT**: Do NOT explicitly state the user's star sign as a fact (e.g., 'You are a Leo'). Instead, subtly weave it in.\n"
            "- **BUDGET**: If budget is not known, ASK for it.\n"
            f"\nUser Message: {request.message}"
        )

        query_payload = {
            "class_method": "query",
            "input": {
                "input": {
                    "message": final_input_message, 
                    "user_id": request.user_id,
                    "history": request.history # Pass the full history!
                }
            }
        }
        
        token = get_auth_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Sending contextualized query to Cloud Agent: {AGENT_ID}")
        response = requests.post(REASONING_ENGINE_URL, json=query_payload, headers=headers)
        
        if response.status_code != 200:
            logger.warning(f"Agent Engine failed params ({response.status_code}), switching to Fallback Gemini: {response.text}")
            # Fallback: Use generic Gemini model directly for the demo
            from vertexai.generative_models import GenerativeModel
            import vertexai
            try:
                vertexai.init(project=PROJECT_ID, location=LOCATION)
                model = GenerativeModel("gemini-2.5-flash-lite")
                # Send the clean message (without critical instruction overhead for raw model)
                fallback_chat = model.start_chat()
                fallback_resp = fallback_chat.send_message(request.message)
                return ChatResponse(response=fallback_resp.text, session_id=request.user_id)
            except Exception as e2:
                raise Exception(f"Fallback failed too: {str(e2)}")
            
        resp_json = response.json()
        
        # Parse Response
        # The agent returns {"output": ...}
        agent_output = resp_json.get('output', "No response.")
        
        # If output is nested (common with reasoning engines), try to extract text
        if isinstance(agent_output, dict):
             agent_output = agent_output.get('output', str(agent_output))
             
        return ChatResponse(response=str(agent_output), session_id=request.user_id)

    except Exception as e:
        logger.error(f"Error during chat: {e}")
        mock_response = (
            "✨ **Cosmic Connection Issue** ✨\n\n"
            "The stars are a bit cloudy (Agent functionality is limited). \n"
            f"Error: {str(e)}\n\n"
            "Try again later! 🌌"
        )
        return ChatResponse(response=mock_response, session_id="error")

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
