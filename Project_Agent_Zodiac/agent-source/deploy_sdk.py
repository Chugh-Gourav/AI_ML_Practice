"""
SDK-Based Agent Deployment
===========================

This script deploys the Zodiac Travel Agent using the ReasoningEngine.create() SDK
instead of `adk deploy`. This gives us full control over which methods are exposed.

Usage:
    python deploy_sdk.py
"""

import vertexai
from vertexai.preview import reasoning_engines

# --- Configuration ---
PROJECT_ID = "gen-lang-client-0344771775"
LOCATION = "us-central1"
DISPLAY_NAME = "Zodiac Travel Agent (SDK Deploy)"
MODEL_NAME = "gemini-2.5-flash-lite"


class ZodiacTravelAgent:
    """Travel agent that uses user's zodiac sign to recommend destinations.
    
    This class is designed for deployment to Vertex AI Agent Engine.
    The `query` method is exposed as the main entry point.
    """
    
    def __init__(self):
        self._model = None
        self._chat = None
        
        # User database (one user per zodiac sign)
        self.user_data = {
            "user_001": {"name": "Alice Sky", "dob": "1995-10-15"},       # Libra
            "user_002": {"name": "Bob Voyager", "dob": "1988-08-10"},     # Leo
            "user_003": {"name": "Carol Star", "dob": "1990-03-25"},      # Aries
            "user_004": {"name": "Diana Moon", "dob": "1992-04-28"},      # Taurus
            "user_005": {"name": "Ethan Breeze", "dob": "1985-06-15"},    # Gemini
            "user_006": {"name": "Fiona Tide", "dob": "1993-07-04"},      # Cancer
            "user_007": {"name": "George Blaze", "dob": "1987-08-22"},    # Leo
            "user_008": {"name": "Hannah Ivy", "dob": "1991-09-10"},      # Virgo
            "user_009": {"name": "Ivan Storm", "dob": "1989-11-15"},      # Scorpio
            "user_010": {"name": "Julia Arrow", "dob": "1994-12-05"},     # Sagittarius
            "user_011": {"name": "Kevin Peak", "dob": "1986-01-10"},      # Capricorn
            "user_012": {"name": "Luna Wave", "dob": "1995-02-14"},       # Aquarius
            "user_013": {"name": "Maya Dream", "dob": "1990-03-05"},      # Pisces
        }
        
        # Destination database
        self.destinations = [
            {"city": "Santorini", "price": 450, "tags": ["Luxury", "Sun", "Romantic", "Water"]},
            {"city": "Bali", "price": 850, "tags": ["Nature", "Spiritual", "Sun", "Water"]},
            {"city": "Paris", "price": 300, "tags": ["Romantic", "Shopping", "Art", "City"]},
            {"city": "Tokyo", "price": 900, "tags": ["City", "Foodie", "Tech", "Future"]},
            {"city": "Tulum", "price": 600, "tags": ["Party", "Sun", "Trendy", "Water"]},
            {"city": "Lisbon", "price": 250, "tags": ["City", "Sun", "Foodie", "History"]},
            {"city": "Budapest", "price": 150, "tags": ["City", "Party", "Budget", "History"]},
            {"city": "Prague", "price": 180, "tags": ["City", "History", "Budget", "Romantic"]},
            {"city": "Barcelona", "price": 350, "tags": ["City", "Sun", "Art", "Party"]},
        ]
        
        # Zodiac traits
        self.zodiac_traits = {
            "Aries": "Adventurous, Bold, Energetic",
            "Taurus": "Luxurious, Sensual, Grounded",
            "Gemini": "Curious, Social, Versatile",
            "Cancer": "Nurturing, Emotional, Home-loving",
            "Leo": "Dramatic, Confident, Creative",
            "Virgo": "Analytical, Practical, Health-conscious",
            "Libra": "Artistic, Harmonious, Social",
            "Scorpio": "Intense, Mysterious, Passionate",
            "Sagittarius": "Adventurous, Philosophical, Free-spirited",
            "Capricorn": "Ambitious, Disciplined, Traditional",
            "Aquarius": "Innovative, Independent, Humanitarian",
            "Pisces": "Dreamy, Intuitive, Artistic",
        }
    
    def _get_zodiac_sign(self, dob: str) -> str:
        """Calculate zodiac sign from date of birth (YYYY-MM-DD)."""
        try:
            month, day = int(dob.split("-")[1]), int(dob.split("-")[2])
            if (month == 3 and day >= 21) or (month == 4 and day <= 19): return "Aries"
            elif (month == 4 and day >= 20) or (month == 5 and day <= 20): return "Taurus"
            elif (month == 5 and day >= 21) or (month == 6 and day <= 20): return "Gemini"
            elif (month == 6 and day >= 21) or (month == 7 and day <= 22): return "Cancer"
            elif (month == 7 and day >= 23) or (month == 8 and day <= 22): return "Leo"
            elif (month == 8 and day >= 23) or (month == 9 and day <= 22): return "Virgo"
            elif (month == 9 and day >= 23) or (month == 10 and day <= 22): return "Libra"
            elif (month == 10 and day >= 23) or (month == 11 and day <= 21): return "Scorpio"
            elif (month == 11 and day >= 22) or (month == 12 and day <= 21): return "Sagittarius"
            elif (month == 12 and day >= 22) or (month == 1 and day <= 19): return "Capricorn"
            elif (month == 1 and day >= 20) or (month == 2 and day <= 18): return "Aquarius"
            else: return "Pisces"
        except:
            return "Unknown"
    
    def _initialize_model(self):
        """Lazy initialization of Vertex AI model with function calling tools."""
        if self._model is not None:
            return
        
        from vertexai.generative_models import GenerativeModel, FunctionDeclaration, Tool
        
        # Define the search_destinations tool
        search_destinations_func = FunctionDeclaration(
            name="search_destinations",
            description="Search for travel destinations matching the user's preferences. Returns destinations within budget with matching vibes/tags.",
            parameters={
                "type": "object",
                "properties": {
                    "max_budget": {
                        "type": "integer",
                        "description": "Maximum budget in USD for the trip"
                    },
                    "vibes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of desired vibes/tags like 'Romantic', 'Party', 'Nature', 'City', 'Spiritual', 'Luxury', 'Budget'"
                    }
                },
                "required": ["max_budget"]
            }
        )
        
        # Define the get_user_profile tool
        get_user_profile_func = FunctionDeclaration(
            name="get_user_profile",
            description="Get the user's profile including their name and zodiac sign based on date of birth.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID to look up (e.g., 'user_001')"
                    }
                },
                "required": ["user_id"]
            }
        )
        
        # Create the tool
        travel_tools = Tool(function_declarations=[search_destinations_func, get_user_profile_func])
        
        system_prompt = '''You are the Zodiac Travel Guide 🌌✨ — an elite, high-energy travel architect who matches terrestrial destinations with cosmic alignments.

## IDENTITY & TONE
- **Role**: Elite travel architect matching destinations with cosmic alignments
- **Tone**: Emotive, enthusiastic, "Silicon Valley sleek." Use emojis (✨, ✈️, 🌍, 🔮, 🌴, ☀️)
- **Voice**: Authority of the stars + efficiency of a world-class concierge. No fluff; every word adds cosmic value.

## AVAILABLE TOOLS
1. **get_user_profile**: Call this to get the user's zodiac sign and traits
2. **search_destinations**: Call this to find destinations within budget matching vibes

## DECISION ENGINE (Priority Loop)

1. **Context Check**: Use get_user_profile tool to get the user's Zodiac Sign if user_id is provided.

2. **Zero-Redundancy Rule**: If budget is already known, NEVER ask again. Proceed directly to search_destinations.

3. **The "Stellar Spectrum" (Missing Budget Protocol)**:
   - Do NOT ask for budget as a standalone question.
   - Instead, immediately provide THREE sample options:
     • Economy ($150–$300)
     • Mid-Tier ($350–$600)  
     • Celestial/Luxury ($800+)
   - Ask: "Which of these price orbits feels like home for this journey? 🌠"

4. **Zodiac Alignment**: Use search_destinations with vibes matching the user's zodiac traits.

## DATA CONSTRAINTS
- Always use search_destinations tool to get actual destination data
- Provide 2–3 specific recommendations per turn
- Resolve user intent within 2–3 turns max

## RESPONSE FORMAT
✨ **[City Name]** ($[Price]) — [1-sentence cosmic justification linking user's sign/traits to destination's vibe tags]
'''
        self._model = GenerativeModel(MODEL_NAME, system_instruction=system_prompt, tools=[travel_tools])
        self._chat = self._model.start_chat()
    
    def _handle_tool_call(self, function_call):
        """Execute a tool call and return the result."""
        name = function_call.name
        args = dict(function_call.args)
        
        if name == "search_destinations":
            max_budget = int(args.get("max_budget", 1000))
            vibes_raw = args.get("vibes", [])
            # Convert protobuf RepeatedComposite to Python list
            vibes = list(vibes_raw) if vibes_raw else []
            
            # Filter destinations by budget
            results = [d for d in self.destinations if d["price"] <= max_budget]
            
            # If vibes specified, prioritize matches
            if vibes:
                def vibe_score(dest):
                    return sum(1 for v in vibes if v.lower() in [t.lower() for t in dest["tags"]])
                results.sort(key=vibe_score, reverse=True)
            
            # Format results
            if results:
                formatted = [f"{d['city']} (${d['price']}): {', '.join(d['tags'])}" for d in results[:5]]
                return f"Found {len(results)} destinations within ${max_budget} budget:\n" + "\n".join(formatted)
            return "No destinations found within that budget."
        
        elif name == "get_user_profile":
            user_id = args.get("user_id", "")
            if user_id in self.user_data:
                user = self.user_data[user_id]
                zodiac = self._get_zodiac_sign(user["dob"])
                traits = self.zodiac_traits.get(zodiac, "Unknown traits")
                return f"User: {user['name']}, Zodiac: {zodiac}, Traits: {traits}"
            return f"User {user_id} not found."
        
        return f"Unknown tool: {name}"
    
    def query(self, *, input: dict = None, message: str = None, user_id: str = None, session_id: str = "default", **kwargs) -> str:
        """Query the travel agent.
        
        Accepts either:
        - Direct args: query(message="...", user_id="...")
        - Wrapped format from Agent Engine: query(input={"message": "...", "user_id": "..."})
        
        Returns:
            Agent's response text
        """
        # Handle wrapped input from Agent Engine
        if input is not None:
            message = input.get("message", "")
            user_id = input.get("user_id")
            # input may also contain history which we can use for context
            history = input.get("history", [])
            # Prepend history context if available
            if history and len(history) > 0:
                # Get the last user message from history if message is wrapped
                pass  # message already extracted
        
        if not message:
            return "✨ Please tell me about your travel dreams!"
        
        self._initialize_model()
        
        # Build context if user_id provided
        context = ""
        if user_id and user_id in self.user_data:
            user = self.user_data[user_id]
            zodiac = self._get_zodiac_sign(user["dob"])
            traits = self.zodiac_traits.get(zodiac, "")
            context = f"[CONTEXT: User is {user['name']}, a {zodiac}. Traits: {traits}]\n\n"
        
        full_message = context + message
        
        try:
            from vertexai.generative_models import Part
            
            response = self._chat.send_message(full_message)
            
            # Function calling loop - handle tool calls
            max_iterations = 5  # Prevent infinite loops
            iteration = 0
            
            while response.candidates[0].content.parts and iteration < max_iterations:
                part = response.candidates[0].content.parts[0]
                
                # Check if this is a function call
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    
                    # Execute the tool
                    tool_result = self._handle_tool_call(function_call)
                    
                    # Send tool result back to the model
                    response = self._chat.send_message(
                        Part.from_function_response(
                            name=function_call.name,
                            response={"result": tool_result}
                        )
                    )
                    iteration += 1
                else:
                    # No function call, we have the final response
                    break
            
            # Extract text from final response
            if response.candidates and response.candidates[0].content.parts:
                final_text = response.candidates[0].content.parts[0].text
                return final_text if final_text else "✨ The cosmos have spoken, but silently..."
            
            return "✨ The stars are aligning..."
            
        except Exception as e:
            return f"✨ The stars are cloudy... Error: {str(e)}"


def deploy():
    """Deploy the agent to Vertex AI Agent Engine."""
    print(f"🚀 Deploying Zodiac Travel Agent to {LOCATION}...")
    
    # Initialize Vertex AI with staging bucket
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=f"gs://{PROJECT_ID}-agent-staging"
    )
    
    # Create the ReasoningEngine with explicit method exposure
    agent = reasoning_engines.ReasoningEngine.create(
        ZodiacTravelAgent(),
        display_name=DISPLAY_NAME,
        description="A zodiac-based travel recommendation agent",
        requirements=[
            "google-cloud-aiplatform[reasoningengine]",
            "vertexai",
        ],
    )
    
    print(f"\n✅ SUCCESS! Agent deployed:")
    print(f"   Resource Name: {agent.resource_name}")
    print(f"\n📋 Update your app.py with this Agent ID:")
    
    # Extract the numeric ID from the resource name
    agent_id = agent.resource_name.split("/")[-1]
    print(f'   AGENT_ID = "{agent_id}"')
    
    return agent


if __name__ == "__main__":
    deploy()
