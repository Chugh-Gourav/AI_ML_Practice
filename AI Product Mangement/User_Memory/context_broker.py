import json
import time
import random
from dataclasses import asdict
import mlflow
import llm_client
from models import TravelAgentProfile, AgentState



class ContextBroker:
    def __init__(self, profiles: list[TravelAgentProfile]):
        # Map profiles for quick lookup
        self.profiles_db = {p.agent_id: p for p in profiles}
        self.sessions_db: dict[str, AgentState] = {p.agent_id: AgentState(agent_id=p.agent_id, current_state="IDLE") for p in profiles}
        
        # Redis Cache Simulation: maps agent_id -> (StepCached, TokenJSONString, ExpiryTimestamp)
        # Represents an async real-time feature store with a 3600s TTL
        self.redis_cache: dict[str, tuple[int, str, float]] = {}
        
        # Performance logging
        self.cache_hits = 0
        self.cache_misses = 0
        self.db_read_ops = 0
        self.db_write_ops = 0
        self.total_personalization_audits = 0
        self.privacy_leaks_detected = 0
        self.cookie_stitch_count = 0
        self.latencies = []

    def get_hce_token(self, agent_id: str, step: int, current_vertical: str) -> tuple[dict, float]:
        """
        Retrieves the 3-layer HCE Token for the agent.
        Simulates cache hits (<2ms) and cache misses with DB queries and cookie stitching (~50ms).
        """
        profile = self.profiles_db[agent_id]
        state = self.sessions_db[agent_id]
        
        # Privacy Guardrail Check
        self.total_personalization_audits += 1
        if profile.test_group == "Treatment" and not profile.consent_opt_in:
            if state.cookie_stitched:
                state.consent_compliance_violated = True
                self.privacy_leaks_detected += 1
        
        # Check cache
        cache_key = agent_id
        is_cache_hit = False
        
        if cache_key in self.redis_cache:
            cached_step, token_str, expiry = self.redis_cache[cache_key]
            # Handle Staleness / TTL expiry explicitly
            if time.time() < expiry and step - cached_step <= 1:
                is_cache_hit = True
                
        if is_cache_hit:
            self.cache_hits += 1
            latency = random.uniform(5.0, 15.0) # Realistic VPC P99 latency
            token = json.loads(token_str)
        else:
            self.cache_misses += 1
            self.db_read_ops += 1
            latency = random.uniform(40.0, 60.0)
            
            if profile.consent_opt_in:
                state.cookie_stitched = True
                self.cookie_stitch_count += 1
            else:
                state.cookie_stitched = False
                
            token = {
                "core_identity": {
                    "origin_market": profile.origin,
                    "device": profile.device,
                    "login_status": profile.login_status,
                    "consent": profile.consent_opt_in,
                    "personalisation_id": profile.personalisation_id if profile.consent_opt_in else "ANONYMIZED"
                },
                "operational_preferences": {
                    "baggage_tolerance": profile.baggage_tolerance if (profile.consent_opt_in or profile.login_status == "Logged In") else "default",
                    "proximity_anxiety": profile.proximity_anxiety if (profile.consent_opt_in or profile.login_status == "Logged In") else "default",
                    "preferred_alliance": profile.preferred_alliance if (profile.consent_opt_in or profile.login_status == "Logged In") else "none"
                },
                "intent_token": {
                    "active_destination": state.active_destination if profile.consent_opt_in else "None",
                    "trip_vibe": state.trip_vibe if profile.consent_opt_in else "None",
                    "flight_dates": state.flight_dates if profile.consent_opt_in else ""
                }
            }
            # Write to Redis with a 1 hour TTL (simulated)
            self.redis_cache[cache_key] = (step, json.dumps(token), time.time() + 3600)
            self.db_write_ops += 1
            
        self.latencies.append(latency)
        return token, latency

    @mlflow.trace(name="intent_extraction", span_type="tool")
    def run_llm_intent_extraction(self, agent_id: str, origin: str, search_query: dict, step: int):
        """Calls Gemini API to extract the initial intent token (destination, dates, vibe) from raw search actions."""
        profile = self.profiles_db[agent_id]
        state = self.sessions_db[agent_id]
        
        if not profile.consent_opt_in:
            # Under mock mode, fallback to standard rule-based setting
            dest = search_query.get("destination", "LON")
            dates = search_query.get("dates", "2026-07-15")
            vibe = profile.persona
            state.active_destination = dest
            state.trip_vibe = vibe
            state.flight_dates = dates
            self.db_write_ops += 1
            return {
                "active_destination": dest,
                "flight_dates": dates,
                "trip_vibe": vibe
            }
            
        print(f"      🤖 [LLM Call {llm_client.usage_stats['llm_calls_made']+1}] Extracting Initial Intent for {agent_id}...")
        
        prompt = f"""You are the Context Engine Horizontal Context Engine (Intent Broker). Analyze this search query and extract the traveler's intent:
        Origin: {origin}
        Search Action details: {json.dumps(search_query)}
        Traveler Persona Context: {profile.persona}
        
        Extract and return a JSON object with:
        - "active_destination": airport code (3 letters, e.g. LHR, DEL, BOM, EDI, NYC, LAX)
        - "flight_dates": "YYYY-MM-DD to YYYY-MM-DD" matching their target dates
        - "trip_vibe": inferred traveler vibe, one of ["Value Hacker", "Loyalty Loyalist", "Multi-Gen Family Planner", "Business Bleisure", "Solo Explorer"]
        - "confidence_score": float between 0.0 and 1.0 representing confidence in this inference
        
        Respond only with valid JSON. Do not include markdown blocks or explanation text."""
        
        try:
            response = llm_client.generate_content_with_retry(prompt, response_mime_type="application/json")
            res_json = json.loads(response.text.strip())
            
            state.active_destination = res_json.get("active_destination", "LON")
            state.flight_dates = res_json.get("flight_dates", "2026-07-15 to 2026-07-22")
            state.trip_vibe = res_json.get("trip_vibe", profile.persona)
            confidence = res_json.get("confidence_score", 0.9)
            
            # Simulated Rules Engine: Fallback if confidence is too low
            if confidence < 0.6:
                print(f"      ⚠️ Low confidence ({confidence}). Falling back to Default UI.")
                raise ValueError("Low confidence inference")
                
            self.db_write_ops += 1
            
            # Sync cache
            if agent_id in self.redis_cache:
                _, token_str, expiry = self.redis_cache[agent_id]
                token = json.loads(token_str)
                token["intent_token"]["active_destination"] = state.active_destination
                token["intent_token"]["trip_vibe"] = state.trip_vibe
                token["intent_token"]["flight_dates"] = state.flight_dates
                self.redis_cache[agent_id] = (step, json.dumps(token), expiry)
                
            if llm_client.LIVE_LLM_MODE:
                time.sleep(1.5) # heavier prompt takes more rate limits politeness
            return res_json
            
        except Exception as e:
            print(f"      ⚠️ Gemini Intent Extraction failed: {e}. Falling back gracefully.")
            span = mlflow.get_current_active_span()
            if span:
                span.set_attribute("fallback_triggered", True)
                span.set_attribute("error", str(e))
                
            # Fallback
            dest = search_query.get("destination", "LON")
            dates = search_query.get("dates", "2026-07-15")
            state.active_destination = dest
            state.trip_vibe = profile.persona
            state.flight_dates = dates
            self.db_write_ops += 1
            return {
                "active_destination": dest,
                "flight_dates": dates,
                "trip_vibe": profile.persona
            }

    @mlflow.trace(name="memory_extraction", span_type="tool")
    def run_llm_memory_extraction(self, agent_id: str, clicks: list):
        """Calls Gemini API asynchronously in the background to extract long-term preferences."""
        profile = self.profiles_db[agent_id]
        state = self.sessions_db[agent_id]
        
        if not profile.consent_opt_in:
            # If user hasn't opted in, fallback to no memory processing
            return {}
            
        print(f"      🤖 [LLM Call {llm_client.usage_stats['llm_calls_made']+1}] Extracting Memory for {agent_id}...")
        
        prompt = f"""You are the Context Engine Horizontal Context Engine (Memory Broker). Analyze these session clicks and extract the traveler's long-term preferences:
        Clicks Log: {json.dumps(clicks[-5:])}
        Current Profile: {json.dumps(asdict(profile))}
        
        Extract and return a JSON object with:
        - "preferred_alliance": one of ["star_alliance", "oneworld", "skyteam", "none"]
        - "baggage_tolerance": one of ["carry_on_only", "low_checked", "heavy_checked"]
        - "proximity_anxiety": one of ["high", "low"]
        - "trip_vibe": one of ["Value Hacker", "Loyalty Loyalist", "Multi-Gen Family Planner", "Business Bleisure", "Solo Explorer"]
        
        Respond only with valid JSON. Do not include markdown blocks or explanation text."""
        
        try:
            response = llm_client.generate_content_with_retry(prompt, response_mime_type="application/json")
            res_json = json.loads(response.text.strip())
            
            # Apply LLM extracted and consolidated changes to profile
            profile.preferred_alliance = res_json.get("preferred_alliance", profile.preferred_alliance)
            profile.baggage_tolerance = res_json.get("baggage_tolerance", profile.baggage_tolerance)
            profile.proximity_anxiety = res_json.get("proximity_anxiety", profile.proximity_anxiety)
            self.db_write_ops += 1
            
            # rate limiter politeness
            if llm_client.LIVE_LLM_MODE:
                time.sleep(1.0)
            return res_json
        except Exception as e:
            print(f"      ⚠️ Gemini Extraction failed: {e}. Falling back gracefully.")
            span = mlflow.get_current_active_span()
            if span:
                span.set_attribute("fallback_triggered", True)
                span.set_attribute("error", str(e))
            return {}
