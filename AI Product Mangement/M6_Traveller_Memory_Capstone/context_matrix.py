class ContextExtensionMatrix:
    @staticmethod
    def get_environmental_nuance(origin: str, destination: str, step: int) -> dict:
        nuances = {}
        if destination in ["DEL", "BOM"] and step >= 12:
            nuances["anomaly"] = "Monsoon Season Ending"
            nuances["modifier"] = "flight delays expected, taxi redirects priority"
        elif destination in ["NYC", "LAX"] and step >= 18:
            nuances["anomaly"] = "Currency Surge"
            nuances["modifier"] = "budget warning badge, hostel/LCC swap trigger"
        else:
            nuances["anomaly"] = "None"
            nuances["modifier"] = ""
            
        return nuances
