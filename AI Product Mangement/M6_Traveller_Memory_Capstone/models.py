import random
from dataclasses import dataclass

@dataclass
class TravelAgentProfile:
    agent_id: str
    origin: str          # DEL, BOM, LON, EDI, NYC, LAX
    country: str         # India, UK, US
    device: str          # Mobile App, Desktop Web
    login_status: str     # Logged In, Anonymous
    consent_opt_in: bool
    tracking_id: str
    ads_id: str
    personalisation_id: str
    persona: str         # Value Hacker, Loyalty Loyalist, Multi-Gen Family Planner, Business Bleisure, Solo Explorer
    test_group: str      # Control, Treatment
    
    # Operational preferences
    baggage_tolerance: str  # low_checked, carry_on_only, heavy_checked
    proximity_anxiety: str  # high, low
    preferred_alliance: str # star_alliance, skyteam, oneworld, none

@dataclass
class AgentState:
    agent_id: str
    current_state: str  # IDLE, SEARCHING_FLIGHTS, FLIGHT_REDIRECTED, SEARCHING_STAYS, STAY_REDIRECTED, SEARCHING_TRANSPORT, FULLY_REDIRECTED, ABANDONED
    active_destination: str = "None"
    trip_vibe: str = "None"
    flight_dates: str = ""
    last_active_step: int = 0
    search_count: int = 0
    redirect_count: int = 0
    consecutive_idle_steps: int = 0
    consent_compliance_violated: bool = False
    cookie_stitched: bool = False

def generate_simulated_agents(num_agents: int) -> list[TravelAgentProfile]:
    """Generates user profiles for the simulation based on defined distributions."""
    agents = []
    
    origins = [
        ("DEL", "India", 0.20), ("BOM", "India", 0.20),
        ("LON", "UK", 0.15), ("EDI", "UK", 0.15),
        ("NYC", "US", 0.15), ("LAX", "US", 0.15)
    ]
    
    personas = [
        ("Value Hacker", 0.25),
        ("Loyalty Loyalist", 0.25),
        ("Multi-Gen Family Planner", 0.20),
        ("Business Bleisure", 0.15),
        ("Solo Explorer", 0.15)
    ]
    
    for i in range(1, num_agents + 1):
        agent_id = f"AGT-{i:04d}"
        
        # 1. Geographic Origin Distribution
        r_geo = random.random()
        cumulative_geo = 0.0
        origin, country, origin_weight = origins[0]
        for o, c, w in origins:
            cumulative_geo += w
            if r_geo <= cumulative_geo:
                origin, country = o, c
                break
                
        # 2. Device Type Distribution (40% Mobile App, 60% Desktop Web)
        device = "Mobile App" if random.random() < 0.40 else "Desktop Web"
        
        # 3. Login Status (30% Logged In, 70% Anonymous)
        login_status = "Logged In" if random.random() < 0.30 else "Anonymous"
        
        # 4. Consent Guardrails (80% Opt-In, 20% Opt-Out)
        consent_opt_in = random.random() < 0.80
        
        # 5. Core Identifiers Stitching Logic
        tracking_id = f"TRK-{random.randint(100000, 999999)}"
        ads_id = f"ADS-{random.randint(100000, 999999)}"
        personalisation_id = f"PRSN-{random.randint(100000, 999999)}"
        
        # 6. Behavioral Persona Archetypes Distribution
        r_persona = random.random()
        cumulative_persona = 0.0
        persona = personas[0][0]
        for p, w in personas:
            cumulative_persona += w
            if r_persona <= cumulative_persona:
                persona = p
                break
                
        # 7. A/B Test Group Assignment (50% Control, 50% Treatment)
        test_group = "Treatment" if random.random() < 0.50 else "Control"
        
        # 8. Operational Preferences mapping from Persona
        if persona == "Value Hacker":
            baggage_tolerance = "carry_on_only"
            proximity_anxiety = "low"
            preferred_alliance = "none"
        elif persona == "Loyalty Loyalist":
            baggage_tolerance = "low_checked"
            proximity_anxiety = "low"
            preferred_alliance = random.choice(["star_alliance", "oneworld", "skyteam"])
        elif persona == "Multi-Gen Family Planner":
            baggage_tolerance = "heavy_checked"
            proximity_anxiety = "high"
            preferred_alliance = "none"
        elif persona == "Business Bleisure":
            baggage_tolerance = "low_checked"
            proximity_anxiety = "high"
            preferred_alliance = random.choice(["star_alliance", "oneworld", "skyteam"])
        else: # Solo Explorer
            baggage_tolerance = "carry_on_only"
            proximity_anxiety = "low"
            preferred_alliance = "none"
            
        agent = TravelAgentProfile(
            agent_id=agent_id,
            origin=origin,
            country=country,
            device=device,
            login_status=login_status,
            consent_opt_in=consent_opt_in,
            tracking_id=tracking_id,
            ads_id=ads_id,
            personalisation_id=personalisation_id,
            persona=persona,
            test_group=test_group,
            baggage_tolerance=baggage_tolerance,
            proximity_anxiety=proximity_anxiety,
            preferred_alliance=preferred_alliance
        )
        agents.append(agent)
        
    return agents
