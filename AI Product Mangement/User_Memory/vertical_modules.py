class VerticalModules:
    @staticmethod
    def get_flights_layout(token: dict, is_treatment: bool, destination: str) -> str:
        if not is_treatment or not token["core_identity"]["consent"]:
            return "DEFAULT_SORT: Price/Duration Ascending | BANNERS: Generic Skyscanner Promo"
            
        persona = token["intent_token"]["trip_vibe"]
        alliance = token["operational_preferences"]["preferred_alliance"]
        
        layout = "HCE_SORT: Optimized Refactor | "
        
        if alliance != "none":
            layout += f"AllianceHighlight: {alliance} carriers prioritized with Avios/Miles badges | "
        else:
            layout += "StandardCarrierSort | "
            
        if persona == "Value Hacker":
            layout += "LCC_Focus: Surfaced Indigo/Wizz with baggage calculators highlighted"
        elif persona == "Multi-Gen Family Planner":
            layout += "GroupBookingRates: Prominently shows multi-passenger discounts & baggage inclusions"
        elif alliance != "none":
            layout += "CoBrandCardRewards: Earn double points on Chase Sapphire"
        else:
            layout += "StandardOffers"
            
        if destination in ["DEL", "BOM"]:
            layout += " | PaymentPriority: Direct payment via UPI (GPay/PhonePe)"
            
        return layout

    @staticmethod
    def get_stays_layout(token: dict, is_treatment: bool, destination: str) -> str:
        if not is_treatment or not token["core_identity"]["consent"]:
            return "DEFAULT_HOTEL_SORT: Popularity | INPUTS: Empty Dates & Destination"
            
        intent = token["intent_token"]
        persona = intent["trip_vibe"]
        
        layout = ""
        if intent["active_destination"] == destination and intent["flight_dates"]:
            layout += f"PRE-FILL: Destination={destination}, Dates={intent['flight_dates']} | "
        else:
            layout += "INPUTS: Blank | "
            
        layout += "HCE_HOTEL_SORT: "
        if persona == "Multi-Gen Family Planner":
            layout += f"FamilyFilter: Multi-room apartments, pools, and airport transfers near {destination}"
        elif persona == "Business Bleisure":
            layout += "BusinessFilter: Workspaces, High-speed Wi-Fi, and late checkout stays"
        elif persona == "Solo Explorer":
            layout += "SoloFilter: Top social hostels, homestays, and shared spaces"
        else:
            layout += "GenericRefinedSort"
            
        return layout

    @staticmethod
    def get_transport_layout(token: dict, is_treatment: bool, destination: str) -> str:
        if not is_treatment or not token["core_identity"]["consent"]:
            return "DEFAULT_CAR_SORT: Price Ascending"
            
        persona = token["intent_token"]["trip_vibe"]
        anxiety = token["operational_preferences"]["proximity_anxiety"]
        
        if destination in ["LON", "EDI"] and anxiety == "high":
            return "ANXIETY_NUDGE: Heathrow Express/LNER Rail Ticket Redirect featured prominently over car hire"
            
        if persona == "Multi-Gen Family Planner":
            return "CAR_SORT: SUV and Minivan listings with infant-seat add-ons highlighted"
        elif persona == "Solo Explorer":
            return "CAR_SORT: Low-cost fuel-efficient vehicles & public transit guide cards"
        else:
            return "CAR_SORT: Mid-size standard list"
