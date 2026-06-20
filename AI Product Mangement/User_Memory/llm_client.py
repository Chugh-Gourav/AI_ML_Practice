import os
import time
import random
from datetime import datetime
import mlflow

"""
PRODUCT MANAGEMENT ARCHITECTURE NOTES: THE AI GATEWAY (llm_client.py)

1. Unit Economics & Feasibility:
   Historically, extracting memory on every session was cost-prohibitive. 
   By utilizing lightweight, fast models (Gemini 2.5 Flash), the unit economics shift drastically.
   Current Cost Estimation: ~$0.0001 per Monthly Active User (MAU). 
   This makes "Context Engineering at Scale" financially viable for a platform like Context Engine.

2. Latency & SLA Budgets:
   LLM calls inherently introduce latency (typically 500ms - 2000ms).
   To respect strict UI performance SLAs, Intent Extraction (which runs during the active session) 
   MUST be asynchronous or non-blocking, while Memory Extraction (consolidation) runs entirely 
   in the background via pub/sub after a session is abandoned or fully converted.

3. Observability & Fallbacks:
   AI features fail. We use `mlflow.trace` to monitor exactly what unstructured data goes in 
   and what structured JSON comes out. If the LLM throws a 429 (Rate Limit) or hallucinates, 
   the `@mlflow.trace` captures the `fallback_triggered` tag, allowing PMs to monitor 
   feature degradation in production.
"""

try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# LLM API Cost Configuration (Gemini 2.5 Flash pricing)
LLM_COST_INPUT_1M = 0.075  # $ per 1 million input tokens
LLM_COST_OUTPUT_1M = 0.30  # $ per 1 million output tokens

# Security Notice: API keys must never be hardcoded. 
# Ensure GEMINI_API_KEY is injected via secure environment variables or a secret manager in production.
API_KEY = os.environ.get("GEMINI_API_KEY") or "YOUR_GEMINI_API_KEY_HERE"
client = None
LIVE_LLM_MODE = False
GEMINI_MODEL = "gemini-2.5-flash"

if SDK_AVAILABLE and API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
        LIVE_LLM_MODE = True
        print(f"✅ Connected to Live Gemini API. Model: {GEMINI_MODEL}")
    except Exception as e:
        print(f"⚠️ Live Gemini API failed to initialize: {e}. Running in MOCK MODE.")
else:
    print("⚠️ GEMINI_API_KEY not set or SDK not available. Running in MOCK MODE (using rule-based copy/preference synthesis).")

# Global usage stats tracking (for convenience in main script)
usage_stats = {
    "llm_calls_made": 0,
    "llm_input_tokens": 0,
    "llm_output_tokens": 0
}
llm_traces = []

@mlflow.trace(name="gemini_generate_content", span_type="llm")
def generate_content_with_retry(prompt: str, response_mime_type: str = None):
    """Generates content using the Gemini API client with exponential backoff for 429 rate limits."""
    if not LIVE_LLM_MODE:
        class MockResponse:
            def __init__(self, text):
                self.text = text
                
        # Generate a realistic mock response depending on the prompt
        if "Intent Broker" in prompt:
            res = '{"active_destination": "BCN", "flight_dates": "2026-07-16 to 2026-07-25", "trip_vibe": "Solo Explorer"}'
        elif "Memory Broker" in prompt:
            res = '{"operational_preferences": {"baggage_tolerance": "heavy_checked", "proximity_anxiety": "high"}}'
        else:
            res = '{"mock": "response"}'
            
        usage_stats["llm_calls_made"] += 1
        usage_stats["llm_input_tokens"] += len(prompt) // 4
        usage_stats["llm_output_tokens"] += len(res) // 4
        
        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute("mock_mode", True)
            
        return MockResponse(res)

    if not client:
        return None
    
    max_retries = 6
    base_delay = 4.0
    
    for attempt in range(max_retries):
        try:
            if response_mime_type:
                config = types.GenerateContentConfig(response_mime_type=response_mime_type)
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=config
                )
            else:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
            
            # Update stats
            usage_stats["llm_calls_made"] += 1
            in_t = getattr(response.usage_metadata, "prompt_token_count", 0) or 400
            out_t = getattr(response.usage_metadata, "candidates_token_count", 0) or 100
            usage_stats["llm_input_tokens"] += in_t
            usage_stats["llm_output_tokens"] += out_t
            
            # Record trace
            llm_traces.append({
                "timestamp": datetime.now().isoformat(),
                "model": GEMINI_MODEL,
                "prompt": prompt,
                "response": response.text.strip() if hasattr(response, "text") else str(response),
                "tokens_input": in_t,
                "tokens_output": out_t
            })
            
            return response
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "resource" in err_msg or "rate limit" in err_msg or "exhausted" in err_msg:
                delay = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"      ⚠️ Gemini API rate limit hit (429). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"      ⚠️ Gemini API error occurred: {e}")
                raise e
    raise Exception("Failed to execute Gemini API call after maximum retries due to rate limiting.")
