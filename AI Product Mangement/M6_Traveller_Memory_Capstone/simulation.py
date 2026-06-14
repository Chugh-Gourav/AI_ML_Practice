"""
Skyscanner Horizontal Context Engine (HCE) - Traveller Memory Simulation
Module 6: Capstone Project

This script programmatically simulates travelers over a 24-step sequence
to validate a centralized context/memory layer (HCE) across Flights, Stays, and Transport.
It integrates a live Google GenAI SDK (Gemini 2.0 Flash) for background memory extraction,
preference consolidation, and Price Drop copy generation.
It compares a personalized Treatment group with a siloed Control group, monitors privacy compliance,
evaluates database and caching latencies, estimates scaling costs, and exports
four deliverables (3 CSVs + 1 interactive HTML playground).

This script is written in a modular fashion, importing components from:
- models.py (for TravelAgentProfile, AgentState, and generate_simulated_agents)
- context_broker.py (for ContextBroker state management and LLM extraction routing)
- vertical_modules.py (for VerticalModules layout logic)
- context_matrix.py (for environmental anomalies)
- visualizer.py (for interactive HTML rendering)
- llm_client.py (for live Gemini SDK connectivity and backoff retry logic)

PRODUCT MANAGEMENT ARCHITECTURE NOTES: THE EVENT ORCHESTRATOR (simulation.py)

1. The "Multi-Tab Comparison Shopper" Edge Case:
   Traditional SQL-based funnel analytics (Google Analytics, Mixpanel) assume a linear state machine 
   (e.g., Home -> Search -> Redirect). In travel, users "cmd+click" multiple tabs, causing chronologically 
   overlapping, non-linear events. By passing the raw JSON "bag of clicks" to an LLM, the Context Engine 
   can infer complex behaviors (like comparison shopping) without breaking, eliminating the need for rigid SQL.

2. Rich Action Context:
   Notice that `log_interaction` was updated to include specific `action_details` (like filters clicked).
   If you want an LLM to infer long-term memory (e.g., "baggage tolerance = carry-on only"), it cannot 
   guess out of thin air. The frontend event stream must explicitly capture the granular context of the UI.
"""

import os
import csv
import math
import time
import random
from datetime import datetime
import mlflow

# Import modular components
from models import TravelAgentProfile, AgentState, generate_simulated_agents
from context_broker import ContextBroker
from vertical_modules import VerticalModules
from context_matrix import ContextExtensionMatrix
from visualizer import generate_html_visualizer
import llm_client

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
OUTPUT_DIR = "/Users/gouravsstudy/Desktop/AI Revision, and Fun Learning/AI Product Mangement/M6_Traveller_Memory_Capstone"
NUM_AGENTS = 30
STEPS = 24
COMMUNICATION_LIMIT = 20

# Random seed for reproducibility
random.seed(42)

class TravellerMemorySimulation:
    def __init__(self):
        # Generate population using the models module
        self.agents = generate_simulated_agents(NUM_AGENTS)
        # Initialize context broker using the context_broker module
        self.broker = ContextBroker(self.agents)
        
        # Interaction logs & alerts registers
        self.interaction_logs = []
        self.communications_logs = []
        self.communications_sent = 0
        
        self.searches = {"Control": 0, "Treatment": 0}
        self.redirects = {"Control": 0, "Treatment": 0}
        self.cross_vertical_redirects = {"Control": 0, "Treatment": 0}
        
        self.comms_clicks = {"Control": 0, "Treatment": 0}
        self.comms_conversions = {"Control": 0, "Treatment": 0}

    def run(self):
        import mlflow
        mlflow.set_experiment("Skyscanner_HCE_Simulation")
        mlflow.start_run(run_name=f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            "num_agents": NUM_AGENTS,
            "steps": STEPS,
            "communication_limit": COMMUNICATION_LIMIT,
            "llm_model": llm_client.GEMINI_MODEL,
            "llm_mode": "LIVE" if llm_client.LIVE_LLM_MODE else "MOCK"
        })
        
        destinations = ["DEL", "BOM", "LON", "EDI", "NYC", "LAX"]
        
        print(f"🚀 Starting {NUM_AGENTS} Traveller Memory Agent-Based Simulation...")
        print(f"   Steps (days): {STEPS} | Population: {NUM_AGENTS} | Comm Limit: {COMMUNICATION_LIMIT}")
        print(f"   LLM Mode: {'LIVE (Gemini 2.0 Flash)' if llm_client.LIVE_LLM_MODE else 'MOCK (Rule-based)'}")
        
        for step in range(1, STEPS + 1):
            for agent in self.agents:
                state = self.broker.sessions_db[agent.agent_id]
                
                if state.current_state in ["ABANDONED", "FULLY_REDIRECTED"]:
                    continue
                    
                is_treatment = (agent.test_group == "Treatment")
                token, latency = self.broker.get_hce_token(agent.agent_id, step, "Flights")
                action = self.decide_agent_action(agent, state, is_treatment, token)
                
                if action == "SEARCH":
                    state.search_count += 1
                    self.searches[agent.test_group] += 1
                    
                    if state.current_state == "IDLE":
                        state.current_state = "SEARCHING_FLIGHTS"
                        dest = random.choice([d for d in destinations if d != agent.origin[:3]])
                        dates_start = random.randint(10, 20)
                        dates_str = f"2026-07-{dates_start} to 2026-07-{dates_start+7}"
                        
                        search_payload = {
                            "destination": dest,
                            "dates": dates_str,
                            "passengers": 1 if agent.persona in ["Solo Explorer", "Business Bleisure"] else (4 if agent.persona == "Multi-Gen Family Planner" else 2)
                        }
                        
                        # Call LLM Intent Extraction (for Treatment)
                        if is_treatment and agent.consent_opt_in:
                            self.broker.run_llm_intent_extraction(agent.agent_id, agent.origin, search_payload, step)
                        else:
                            # Direct database write for Control / Opt-out
                            state.active_destination = dest
                            state.flight_dates = dates_str
                            state.trip_vibe = agent.persona if agent.consent_opt_in else "None"
                            self.broker.db_write_ops += 1
                    elif state.current_state == "FLIGHT_REDIRECTED":
                        state.current_state = "SEARCHING_STAYS"
                    elif state.current_state == "STAY_REDIRECTED":
                        state.current_state = "SEARCHING_TRANSPORT"
                        
                    layout = self.get_applied_layout(state.current_state, token, is_treatment, state.active_destination)
                    details = {"destination": state.active_destination, "dates": state.flight_dates}
                    self.log_interaction(step, agent, state, "SEARCH", layout, latency, details)
                    state.consecutive_idle_steps = 0
                    
                elif action == "NAVIGATE_TAB":
                    if state.current_state == "SEARCHING_FLIGHTS":
                        state.current_state = "SEARCHING_STAYS"
                    elif state.current_state == "SEARCHING_STAYS":
                        state.current_state = "SEARCHING_TRANSPORT"
                    else:
                        state.current_state = "SEARCHING_FLIGHTS"
                        
                    if is_treatment and agent.consent_opt_in:
                        self.cross_vertical_redirects[agent.test_group] += 1
                        
                    layout = self.get_applied_layout(state.current_state, token, is_treatment, state.active_destination)
                    details = {"target_vertical": state.current_state}
                    self.log_interaction(step, agent, state, "NAVIGATE_TAB", layout, latency, details)
                    state.consecutive_idle_steps = 0
                    
                elif action == "REDIRECT_PARTNER":
                    state.redirect_count += 1
                    self.redirects[agent.test_group] += 1
                    
                    if state.current_state == "SEARCHING_STAYS":
                        self.cross_vertical_redirects[agent.test_group] += 1
                        state.current_state = "STAY_REDIRECTED"
                    elif state.current_state == "SEARCHING_TRANSPORT":
                        self.cross_vertical_redirects[agent.test_group] += 1
                        state.current_state = "FULLY_REDIRECTED"
                    else:
                        state.current_state = "FLIGHT_REDIRECTED"
                        
                    layout = self.get_applied_layout(state.current_state, token, is_treatment, state.active_destination)
                    
                    # Log the specific filters the user interacted with before redirecting
                    filters = []
                    if agent.baggage_tolerance == "carry_on_only": filters.append("Carry-on Included")
                    if agent.baggage_tolerance == "heavy_checked": filters.append("2x Checked Bags")
                    if agent.preferred_alliance != "none": filters.append(f"Alliance: {agent.preferred_alliance.replace('_', ' ').title()}")
                    if agent.proximity_anxiety == "high": filters.append("Direct Flights Only")
                    
                    details = {
                        "partner_redirected_to": "Booking.com" if "STAY" in state.current_state else "SkyscannerPartner",
                        "applied_filters": filters
                    }
                    self.log_interaction(step, agent, state, "REDIRECT_PARTNER", layout, latency, details)
                    
                    # Background Memory Extraction & Consolidation (LLM Call)
                    if is_treatment and agent.consent_opt_in:
                        agent_clicks = [l for l in self.interaction_logs if l["agent_id"] == agent.agent_id]
                        self.broker.run_llm_memory_extraction(agent.agent_id, agent_clicks)
                        
                    state.consecutive_idle_steps = 0
                    
                elif action == "ABANDON":
                    if state.search_count >= 2 and state.redirect_count == 0:
                        state.current_state = "ABANDONED"
                        self.log_interaction(step, agent, state, "ABANDON", "N/A", latency)
                    else:
                        state.consecutive_idle_steps += 1
                        if state.consecutive_idle_steps >= 4:
                            state.current_state = "ABANDONED"
                            self.log_interaction(step, agent, state, "ABANDON", "N/A", latency)
                        else:
                            self.log_interaction(step, agent, state, "IDLE", "N/A", latency)
                            
            # Trigger communications step
            self.trigger_communications(step)

        print("✅ Simulation complete! Processing scorecard and generating deliverables...")
        # self.generate_csv_deliverables()  # Commented out to keep directory clean
        # Delegate HTML generation to visualizer.py
        generate_html_visualizer(self.agents, self.interaction_logs, self.communications_logs, OUTPUT_DIR)
        self.print_dashboard()
        self.log_mlflow_data()
        mlflow.end_run()

    def decide_agent_action(self, agent: TravelAgentProfile, state: AgentState, is_treatment: bool, token: dict) -> str:
        p_search = 0.40
        p_tab = 0.15
        p_redirect = 0.10
        p_abandon = 0.35
        
        if is_treatment and agent.consent_opt_in:
            if state.current_state == "IDLE":
                p_search = 0.55
                p_abandon = 0.30
            elif state.current_state == "SEARCHING_FLIGHTS":
                p_redirect = 0.22
                p_tab = 0.25
                p_abandon = 0.20
            elif state.current_state == "FLIGHT_REDIRECTED":
                p_tab = 0.35
                p_abandon = 0.25
            elif state.current_state == "SEARCHING_STAYS":
                p_redirect = 0.20
                p_tab = 0.15
                p_abandon = 0.25
            elif state.current_state == "STAY_REDIRECTED":
                p_tab = 0.40
                p_abandon = 0.20
            elif state.current_state == "SEARCHING_TRANSPORT":
                p_redirect = 0.25
                p_abandon = 0.20
        else:
            if state.current_state == "FLIGHT_REDIRECTED":
                p_tab = 0.08
                p_abandon = 0.60
            elif state.current_state == "STAY_REDIRECTED":
                p_tab = 0.05
                p_abandon = 0.70

        if agent.persona == "Business Bleisure":
            p_redirect *= 1.3
            p_abandon *= 0.7
        elif agent.persona == "Value Hacker":
            p_redirect *= 0.7
            p_abandon *= 1.3
        elif agent.persona == "Multi-Gen Family Planner":
            p_tab *= 1.2
            
        r = random.random()
        total_p = p_search + p_tab + p_redirect + p_abandon
        norm_search = p_search / total_p
        norm_tab = p_tab / total_p
        norm_redirect = p_redirect / total_p
        
        if r <= norm_search:
            return "SEARCH"
        elif r <= (norm_search + norm_tab):
            return "NAVIGATE_TAB"
        elif r <= (norm_search + norm_tab + norm_redirect):
            return "REDIRECT_PARTNER"
        else:
            return "ABANDON"

    def get_applied_layout(self, current_state: str, token: dict, is_treatment: bool, destination: str) -> str:
        if current_state in ["SEARCHING_FLIGHTS", "FLIGHT_REDIRECTED"]:
            return VerticalModules.get_flights_layout(token, is_treatment, destination)
        elif current_state in ["SEARCHING_STAYS", "STAY_REDIRECTED"]:
            return VerticalModules.get_stays_layout(token, is_treatment, destination)
        elif current_state in ["SEARCHING_TRANSPORT", "FULLY_REDIRECTED"]:
            return VerticalModules.get_transport_layout(token, is_treatment, destination)
        return "Default Home UI Layout"

    def log_interaction(self, step: int, agent: TravelAgentProfile, state: AgentState, action: str, layout: str, latency: float, action_details: dict = None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "agent_id": agent.agent_id,
            "test_group": agent.test_group,
            "action": action,
            "state_entered": state.current_state,
            "applied_ui_layout": layout,
            "api_latency_ms": f"{latency:.2f}",
            "consent_opt_in": agent.consent_opt_in,
            "cookie_stitched": state.cookie_stitched
        }
        if action_details:
            log_entry["action_details"] = action_details
        self.interaction_logs.append(log_entry)

    @mlflow.trace(name="generate_price_drop_push", span_type="tool")
    def generate_push_notification(self, agent, dest, nuance_str):
        print(f"      🤖 [LLM Call {llm_client.usage_stats['llm_calls_made']+1}] Generating alert copy for {agent.agent_id}...")
        prompt = f"""You are a growth marketing copywriter at Skyscanner. Write a short, highly compelling Price Drop push notification (maximum 120 characters) for a traveler.
        - Destination: {dest}
        - Persona: {agent.persona}
        - Origin Market: {agent.origin}
        - Preferred Alliance: {agent.preferred_alliance}
        - Environmental Modifier: {nuance_str}
        
        Guidelines:
        1. Avoid templates.
        2. Do not offer cashbacks.
        3. Respond only with the alert text under 120 characters."""
        
        try:
            response = llm_client.generate_content_with_retry(prompt)
            payload = response.text.strip()
            if payload.startswith('"') and payload.endswith('"'):
                payload = payload[1:-1]
            time.sleep(0.5)
            
            span = mlflow.get_current_active_span()
            if span:
                span.set_attribute("agent_id", agent.agent_id)
                span.set_attribute("destination", dest)
                span.set_attribute("persona", agent.persona)
            return {"push_text": payload}
        except Exception as e:
            print(f"      ⚠️ Gemini alert generation failed: {e}. Using fallback.")
            span = mlflow.get_current_active_span()
            if span:
                span.set_attribute("fallback_triggered", True)
                span.set_attribute("error", str(e))
            return {}

    def trigger_communications(self, step: int):
        candidates = []
        for agent in self.agents:
            state = self.broker.sessions_db[agent.agent_id]
            if state.current_state == "ABANDONED" and state.search_count >= 2 and state.redirect_count == 0:
                already_comm = any(c["agent_id"] == agent.agent_id for c in self.communications_logs)
                if not already_comm:
                    candidates.append(agent)
                    
        target_step_comms = int(math.ceil((COMMUNICATION_LIMIT - self.communications_sent) / (STEPS - step + 1)))
        target_step_comms = min(target_step_comms, len(candidates))
        
        if target_step_comms <= 0:
            return
            
        selected_agents = random.sample(candidates, target_step_comms)
        
        for agent in selected_agents:
            state = self.broker.sessions_db[agent.agent_id]
            is_treatment = (agent.test_group == "Treatment")
            
            # Formulate payload
            dest = state.active_destination if state.active_destination != "None" else "London"
            nuances = ContextExtensionMatrix.get_environmental_nuance(agent.origin, dest, step)
            nuance_str = nuances.get("anomaly", "None") + (" (" + nuances.get("modifier") + ")" if nuances.get("modifier") else "")
            
            payload = None
            if is_treatment and agent.consent_opt_in and llm_client.LIVE_LLM_MODE:
                push_res = self.generate_push_notification(agent, dest, nuance_str)
                payload = push_res.get("push_text")
                    
            if not payload:
                # Rule-based fallback alerts
                if is_treatment and agent.consent_opt_in:
                    if agent.persona == "Value Hacker":
                        payload = f"Price Alert: {dest} flights dropped by ₹1,500! Partner airline rates now include free hand luggage."
                    elif agent.persona == "Loyalty Loyalist":
                        payload = f"Price Alert: Partner flight fares to {dest} dropped. Earn double alliance points if you book via Skyscanner partner links today!"
                    elif agent.persona == "Multi-Gen Family Planner":
                        payload = f"Price Alert: {dest} family flights dropped. Lowest partner rates now group passenger seats together automatically."
                    else:
                        payload = f"Price Alert: Partner flight deals to {dest} dropped. Compare rates in your Skyscanner app."
                else:
                    payload = "Price Alert: Skyscanner found a price drop on your recent search! Come back and compare partner deals."
                
            # Simulate click / redirect action
            ctr_chance = 0.55 if (is_treatment and agent.consent_opt_in) else 0.25
            redirect_chance = 0.40 if (is_treatment and agent.consent_opt_in) else 0.15
            
            if agent.persona == "Value Hacker":
                ctr_chance *= 1.2
                redirect_chance *= 1.1
            elif agent.persona == "Business Bleisure":
                ctr_chance *= 0.8
                redirect_chance *= 0.8
                
            clicked = random.random() < ctr_chance
            converted = clicked and (random.random() < redirect_chance)
            
            self.comms_clicks[agent.test_group] += 1 if clicked else 0
            self.comms_conversions[agent.test_group] += 1 if converted else 0
            
            # Generate realistic deep link URL based on your example
            branch_id = f"15384{random.randint(100000, 999999)}"
            deep_link_url = f"https://www.skyscanner.net/flights/search?entity_id={dest}&utm_source=newsletter&utm_medium=email&utm_campaign=price-drop-alert_{step}&_branch_match_id={branch_id}&_branch_referrer=H4sIAAAAA_{agent.agent_id}_encrypted_context"
            
            if clicked:
                # Connected Experience Use Case: Context Rehydration
                token, latency = self.broker.get_hce_token(agent.agent_id, step, "Flights")
                layout = self.get_applied_layout("SEARCHING_FLIGHTS", token, is_treatment, dest)
                # Note: The platform uses the _branch_referrer token to retrieve the agent_id and rehydrate the token above!
                self.log_interaction(step, agent, state, "RETURN_FROM_COMM_VIA_DEEPLINK", layout, latency)
                state.current_state = "SEARCHING_FLIGHTS"
                
                if converted:
                    state.current_state = "FLIGHT_REDIRECTED"
                    state.redirect_count += 1
                    self.redirects[agent.test_group] += 1
                    self.log_interaction(step, agent, state, "REDIRECT_PARTNER_VIA_COMM", "Price Drop Alert Redirect", 2.0)
                
            self.communications_logs.append({
                "communication_id": f"COMM-{self.communications_sent + 1:03d}",
                "agent_id": agent.agent_id,
                "step": step,
                "origin_market": agent.origin,
                "test_group": agent.test_group,
                "persona": agent.persona,
                "payload": payload,
                "deep_link_url": deep_link_url,
                "clicked": clicked,
                "completed_redirect": converted
            })
            
            self.communications_sent += 1
            if self.communications_sent >= COMMUNICATION_LIMIT:
                break

    def generate_csv_deliverables(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 1. user_profiles.csv
        profiles_path = os.path.join(OUTPUT_DIR, "user_profiles.csv")
        with open(profiles_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "agent_id", "origin", "country", "device", "login_status", 
                "consent_opt_in", "tracking_id", "ads_id", "personalisation_id", 
                "persona", "test_group", "baggage_tolerance", "proximity_anxiety", "preferred_alliance"
            ])
            for agent in self.agents:
                writer.writerow([
                    agent.agent_id, agent.origin, agent.country, agent.device, agent.login_status,
                    agent.consent_opt_in, agent.tracking_id, agent.ads_id, agent.personalisation_id,
                    agent.persona, agent.test_group, agent.baggage_tolerance, agent.proximity_anxiety, agent.preferred_alliance
                ])
                
        # 2. interaction_logs.csv
        logs_path = os.path.join(OUTPUT_DIR, "interaction_logs.csv")
        with open(logs_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "step", "agent_id", "test_group", "action", 
                "state_entered", "applied_ui_layout", "api_latency_ms", "consent_opt_in", "cookie_stitched"
            ])
            for log in self.interaction_logs:
                writer.writerow([
                    log["timestamp"], log["step"], log["agent_id"], log["test_group"], log["action"],
                    log["state_entered"], log["applied_ui_layout"], log["api_latency_ms"],
                    log["consent_opt_in"], log["cookie_stitched"]
                ])
                
        # 3. lifecycle_communications.csv
        comms_path = os.path.join(OUTPUT_DIR, "lifecycle_communications.csv")
        with open(comms_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "communication_id", "agent_id", "step", "origin_market", 
                "test_group", "persona", "payload", "clicked", "completed_redirect"
            ])
            for comm in self.communications_logs:
                writer.writerow([
                    comm["communication_id"], comm["agent_id"], comm["step"], comm["origin_market"],
                    comm["test_group"], comm["persona"], comm["payload"], comm["clicked"], comm["completed_redirect"]
                ])
                
        print(f"📂 CSV files successfully written to {OUTPUT_DIR}")

    def print_dashboard(self):
        con_control_searches = self.searches["Control"]
        con_treatment_searches = self.searches["Treatment"]
        
        con_control_redirects = self.redirects["Control"]
        con_treatment_redirects = self.redirects["Treatment"]
        
        c_search_rate = (con_control_redirects / con_control_searches * 100) if con_control_searches > 0 else 0
        t_search_rate = (con_treatment_redirects / con_treatment_searches * 100) if con_treatment_searches > 0 else 0
        
        lift_conversion = t_search_rate - c_search_rate
        
        attach_control = (self.cross_vertical_redirects["Control"] / con_control_redirects * 100) if con_control_redirects > 0 else 0
        attach_treatment = (self.cross_vertical_redirects["Treatment"] / con_treatment_redirects * 100) if con_treatment_redirects > 0 else 0
        lift_attach = attach_treatment - attach_control
        
        total_control_comms = sum(1 for c in self.communications_logs if c["test_group"] == "Control")
        total_treatment_comms = sum(1 for c in self.communications_logs if c["test_group"] == "Treatment")
        
        ctr_control = (self.comms_clicks["Control"] / total_control_comms * 100) if total_control_comms > 0 else 0
        ctr_treatment = (self.comms_clicks["Treatment"] / total_treatment_comms * 100) if total_treatment_comms > 0 else 0
        
        cvr_control = (self.comms_conversions["Control"] / total_control_comms * 100) if total_control_comms > 0 else 0
        cvr_treatment = (self.comms_conversions["Treatment"] / total_treatment_comms * 100) if total_treatment_comms > 0 else 0
        
        hits = self.broker.cache_hits
        misses = self.broker.cache_misses
        hit_rate = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0
        
        sorted_lats = sorted(self.broker.latencies)
        p95_lat = sorted_lats[int(len(sorted_lats) * 0.95)] if sorted_lats else 0
        p99_lat = sorted_lats[int(len(sorted_lats) * 0.99)] if sorted_lats else 0
        
        total_db_size_kb = len(self.agents) * 2.0
        
        # Financial model calculations
        if llm_client.LIVE_LLM_MODE:
            total_input_tokens = llm_client.usage_stats["llm_input_tokens"]
            total_output_tokens = llm_client.usage_stats["llm_output_tokens"]
        else:
            # Mock estimation matching NUM_AGENTS scale
            total_redirects = con_control_redirects + con_treatment_redirects
            extraction_input_tokens = total_redirects * 450
            extraction_output_tokens = total_redirects * 120
            comm_input_tokens = total_treatment_comms * 200
            comm_output_tokens = total_treatment_comms * 40
            
            total_input_tokens = extraction_input_tokens + comm_input_tokens
            total_output_tokens = extraction_output_tokens + comm_output_tokens
            
        llm_cost_usd = (total_input_tokens / 1_000_000 * llm_client.LLM_COST_INPUT_1M) + (total_output_tokens / 1_000_000 * llm_client.LLM_COST_OUTPUT_1M)
        
        print("\n" + "="*80)
        print("                SKYSCANNER HCE (TRAVELLER MEMORY) PM SCORECARD")
        print("="*80)
        print(f"1. A/B TEST REVENUE & CONVERSION LIFT")
        print(f"   - Total Searches:       Control: {con_control_searches:<6d} | Treatment: {con_treatment_searches:<6d}")
        print(f"   - Total Partner Redirects: Control: {con_control_redirects:<6d} | Treatment: {con_treatment_redirects:<6d}")
        print(f"   - Redirect Conversion Rate: Control: {c_search_rate:.2f}% | Treatment: {t_search_rate:.2f}%")
        print(f"     👉 CONVERSION LIFT:      {lift_conversion:+.2f}% absolute lift")
        print(f"   - Cross-Vertical Attach Rate: Control: {attach_control:.2f}% | Treatment: {attach_treatment:.2f}%")
        print(f"     👉 ATTACH LIFT:          {lift_attach:+.2f}% absolute lift (Flights to Stays/Car Hire)")
        print("-" * 80)
        print(f"2. LIFECYCLE COMMUNICATIONS ROI ({COMMUNICATION_LIMIT} price alerts)")
        print(f"   - Total Alerts Sent:    Control: {total_control_comms:<6d} | Treatment: {total_treatment_comms:<6d}")
        print(f"   - Alert Click Rate (CTR): Control: {ctr_control:.2f}% | Treatment: {ctr_treatment:.2f}%")
        print(f"   - Alert Redirect Rate:    Control: {cvr_control:.2f}% | Treatment: {cvr_treatment:.2f}%")
        print(f"     👉 ALERT REDIRECT LIFT:  {cvr_treatment - cvr_control:+.2f}% absolute conversion lift")
        print("-" * 80)
        print(f"3. SYSTEM PERFORMANCE & CACHING AUDIT")
        print(f"   - Redis Cache Hit Rate:  {hit_rate:.2f}% ({hits} hits, {misses} misses)")
        print(f"   - Simulated API Latency:  p95: {p95_lat:.2f}ms | p99: {p99_lat:.2f}ms")
        print(f"   - Database Operations:    Read IOPS: {self.broker.db_read_ops} | Write IOPS: {self.broker.db_write_ops}")
        print("-" * 80)
        print(f"4. FINANCIAL & COST-TO-SCALE PROJECTION")
        print(f"   - Memory Profile Storage: {total_db_size_kb:.2f} KB (Structured KV)")
        print(f"   - Total LLM Tokens Used:  Input: {total_input_tokens:,} | Output: {total_output_tokens:,}")
        print(f"   - Est. LLM API Cost:      ${llm_cost_usd:.4f} USD (per {NUM_AGENTS} users / 24 days)")
        print(f"     👉 Scaling Cost per MAU: ${(llm_cost_usd / NUM_AGENTS):.6f} USD")
        print("-" * 80)
        print(f"5. PRIVACY & COMPLIANCE GUARDRAIL AUDIT")
        print(f"   - HCE Personalization Audits: {self.broker.total_personalization_audits}")
        print(f"   - Anonymous Cookie Stitchings: {self.broker.cookie_stitch_count}")
        print(f"   - Consent Compliance Violations: {self.broker.privacy_leaks_detected}")
        if self.broker.privacy_leaks_detected == 0:
            print("     👉 STATUS: 🛡️  100% compliant. Zero personalization leaks detected on opted-out users.")
        else:
            print("     👉 STATUS: ⚠️  CRITICAL PRIVACY VIOLATIONS DETECTED.")
        print("="*80 + "\n")

    def log_mlflow_data(self):
        import mlflow
        # Recalculate metrics for logging
        con_control_searches = self.searches["Control"]
        con_treatment_searches = self.searches["Treatment"]
        con_control_redirects = self.redirects["Control"]
        con_treatment_redirects = self.redirects["Treatment"]
        
        c_search_rate = (con_control_redirects / con_control_searches * 100) if con_control_searches > 0 else 0
        t_search_rate = (con_treatment_redirects / con_treatment_searches * 100) if con_treatment_searches > 0 else 0
        lift_conversion = t_search_rate - c_search_rate
        
        attach_control = (self.cross_vertical_redirects["Control"] / con_control_redirects * 100) if con_control_redirects > 0 else 0
        attach_treatment = (self.cross_vertical_redirects["Treatment"] / con_treatment_redirects * 100) if con_treatment_redirects > 0 else 0
        lift_attach = attach_treatment - attach_control
        
        hits = self.broker.cache_hits
        misses = self.broker.cache_misses
        hit_rate = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0
        
        sorted_lats = sorted(self.broker.latencies)
        p95_lat = sorted_lats[int(len(sorted_lats) * 0.95)] if sorted_lats else 0
        
        if llm_client.LIVE_LLM_MODE:
            total_input_tokens = llm_client.usage_stats["llm_input_tokens"]
            total_output_tokens = llm_client.usage_stats["llm_output_tokens"]
        else:
            total_redirects = con_control_redirects + con_treatment_redirects
            total_input_tokens = total_redirects * 450
            total_output_tokens = total_redirects * 120
            
        llm_cost_usd = (total_input_tokens / 1_000_000 * llm_client.LLM_COST_INPUT_1M) + (total_output_tokens / 1_000_000 * llm_client.LLM_COST_OUTPUT_1M)
        
        mlflow.log_metrics({
            "conversion_lift": lift_conversion,
            "attach_lift": lift_attach,
            "cache_hit_rate": hit_rate,
            "p95_latency_ms": p95_lat,
            "total_llm_cost_usd": llm_cost_usd,
            "cost_per_mau_usd": llm_cost_usd / NUM_AGENTS,
            "privacy_violations": self.broker.privacy_leaks_detected
        })
        
        # Log artifacts (logs and traces)
        mlflow.log_dict(llm_client.llm_traces, "llm_traces.json")
        mlflow.log_dict(self.interaction_logs, "interaction_logs.json")
        mlflow.log_dict(self.communications_logs, "lifecycle_communications.json")
        
        print("📊 Successfully logged parameters, metrics, and traces to MLflow staging console!")


if __name__ == "__main__":
    sim = TravellerMemorySimulation()
    sim.run()
