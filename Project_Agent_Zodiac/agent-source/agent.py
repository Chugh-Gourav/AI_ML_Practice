"""
Zodiac Travel Agent - ADK Pattern Implementation
Based on the Kaggle 5-Day Agent Course (Day 3: Memory Management)

This module implements the Zodiac Travel Agent using:
- LlmAgent: Agent definition with name, model, instruction, and tools
- Gemini: Model wrapper for Gemini API
- Runner: Execution handler with session and memory services
- InMemorySessionService: Short-term conversation memory
- InMemoryMemoryService: Long-term knowledge storage
"""

import os
import asyncio

# --- ADK Imports (moved inside conditional blocks to support cloud deployment) ---
# Note: ADK imports are done inside _LOCAL_DEV_MODE block and ZodiacAgentWrapper
# to prevent import-time initialization failures in cloud environments.
# Only `types` is imported at module level for the wrapper class.
try:
    from google.genai import types
except ImportError:
    types = None  # Will be imported in ZodiacAgentWrapper

# --- API Configuration ---
# Using Vertex AI backend (for Cloud deployment)
# These environment variables configure the ADK's Gemini model to use Vertex AI
GOOGLE_CLOUD_PROJECT = "gen-lang-client-0344771775"
GOOGLE_CLOUD_LOCATION = "us-central1"

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"  # Key setting!
os.environ["GOOGLE_CLOUD_PROJECT"] = GOOGLE_CLOUD_PROJECT
os.environ["GOOGLE_CLOUD_LOCATION"] = GOOGLE_CLOUD_LOCATION

# --- Constants ---
APP_NAME = "ZodiacTravelApp"
USER_ID = "default_user"
MODEL_NAME = "gemini-2.5-flash-lite"

# --- Retry Configuration (from Kaggle notebook) ---
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# --- Data ---
FLIGHT_DATA = [
    {"City": "Santorini", "Price": 450, "Tags": "Luxury, Sun, Romantic, Water"},
    {"City": "Bali", "Price": 850, "Tags": "Nature, Spiritual, Sun, Water"},
    {"City": "Paris", "Price": 300, "Tags": "Romantic, Shopping, Art, City"},
    {"City": "Tokyo", "Price": 900, "Tags": "City, Foodie, Tech, Future"},
    {"City": "Tulum", "Price": 600, "Tags": "Party, Sun, Trendy, Water"},
    {"City": "Lisbon", "Price": 250, "Tags": "City, Sun, Foodie, History"},
    {"City": "Budapest", "Price": 150, "Tags": "City, Party, Budget, History"},
    {"City": "Kyoto", "Price": 950, "Tags": "Nature, Culture, Zen, History"},
    {"City": "Amalfi", "Price": 500, "Tags": "Luxury, Sun, Romantic, Foodie"},
    {"City": "New York", "Price": 500, "Tags": "City, Shopping, High-Energy"}
]

USER_DATA = {
    "user_001": {"name": "Alice Sky", "dob": "1995-10-15"},
    "user_002": {"name": "Bob Voyager", "dob": "1988-08-10"}
}

ZODIAC_TRAITS = {
    "Aries": "Adventurous, Bold, Energetic - loves action-packed destinations",
    "Taurus": "Luxurious, Sensual, Grounded - enjoys comfort and fine dining",
    "Gemini": "Curious, Social, Versatile - loves cities with nightlife",
    "Cancer": "Nurturing, Emotional, Home-loving - prefers relaxing beach resorts",
    "Leo": "Dramatic, Confident, Creative - drawn to glamorous destinations",
    "Virgo": "Analytical, Practical, Health-conscious - enjoys wellness retreats",
    "Libra": "Artistic, Harmonious, Social - loves romantic and beautiful places",
    "Scorpio": "Intense, Mysterious, Passionate - attracted to exotic locations",
    "Sagittarius": "Adventurous, Philosophical, Free-spirited - loves exploration",
    "Capricorn": "Ambitious, Disciplined, Traditional - appreciates historic sites",
    "Aquarius": "Innovative, Independent, Humanitarian - drawn to unique experiences",
    "Pisces": "Dreamy, Intuitive, Artistic - loves spiritual and water destinations"
}


# --- Tool Functions (ADK-compatible) ---
def get_user_profile(user_id: str) -> str:
    """Get user profile including zodiac sign based on date of birth.
    
    Args:
        user_id: The unique identifier for the user (e.g., 'user_001')
    
    Returns:
        A string with user name and zodiac sign, or error message if not found.
    """
    user = USER_DATA.get(user_id)
    if not user:
        return f"User {user_id} not found. Available users: {list(USER_DATA.keys())}"
    
    try:
        dob_month = int(user["dob"].split("-")[1])
        dob_day = int(user["dob"].split("-")[2])
        
        if (dob_month == 3 and dob_day >= 21) or (dob_month == 4 and dob_day <= 19): sign = "Aries"
        elif (dob_month == 4 and dob_day >= 20) or (dob_month == 5 and dob_day <= 20): sign = "Taurus"
        elif (dob_month == 5 and dob_day >= 21) or (dob_month == 6 and dob_day <= 20): sign = "Gemini"
        elif (dob_month == 6 and dob_day >= 21) or (dob_month == 7 and dob_day <= 22): sign = "Cancer"
        elif (dob_month == 7 and dob_day >= 23) or (dob_month == 8 and dob_day <= 22): sign = "Leo"
        elif (dob_month == 8 and dob_day >= 23) or (dob_month == 9 and dob_day <= 22): sign = "Virgo"
        elif (dob_month == 9 and dob_day >= 23) or (dob_month == 10 and dob_day <= 22): sign = "Libra"
        elif (dob_month == 10 and dob_day >= 23) or (dob_month == 11 and dob_day <= 21): sign = "Scorpio"
        elif (dob_month == 11 and dob_day >= 22) or (dob_month == 12 and dob_day <= 21): sign = "Sagittarius"
        elif (dob_month == 12 and dob_day >= 22) or (dob_month == 1 and dob_day <= 19): sign = "Capricorn"
        elif (dob_month == 1 and dob_day >= 20) or (dob_month == 2 and dob_day <= 18): sign = "Aquarius"
        else: sign = "Pisces"
    except:
        sign = "Unknown"
    
    return f"Name: {user['name']}, Zodiac Sign: {sign}"


def search_destinations(vibe_keywords: list[str], max_budget: int) -> str:
    """Search for travel destinations matching vibe keywords within budget.
    
    Args:
        vibe_keywords: List of vibes to match (e.g., ['romantic', 'sun', 'luxury'])
        max_budget: Maximum budget in dollars
    
    Returns:
        Matching destinations with prices and tags, or error message.
    """
    try:
        budget = int(max_budget)
    except:
        return f"Invalid budget format: {max_budget}. Please provide a number."
    
    results = []
    for flight in FLIGHT_DATA:
        if flight["Price"] <= budget:
            score = 0
            tags_lower = flight["Tags"].lower()
            for keyword in vibe_keywords:
                if keyword.lower() in tags_lower:
                    score += 1
            if score > 0:
                results.append(f"✈️ {flight['City']} (${flight['Price']}): {flight['Tags']}")
    
    if not results:
        return f"No destinations found matching {vibe_keywords} under ${budget}. Try increasing budget or different vibes."
    
    return "\n".join(results[:5])


def get_zodiac_traits(sign: str) -> str:
    """Get personality traits and travel preferences for a zodiac sign.
    
    Args:
        sign: The zodiac sign (e.g., 'Leo', 'Pisces')
    
    Returns:
        Personality traits and travel preferences for the sign.
    """
    traits = ZODIAC_TRAITS.get(sign.capitalize())
    if traits:
        return f"🔮 {sign} Traits: {traits}"
    return f"Unknown sign: {sign}. Valid signs: {list(ZODIAC_TRAITS.keys())}"


# --- Agent Definition ---
SYSTEM_INSTRUCTION = """You are an ENERGETIC, EMOTIVE, and HELPFUL Zodiac Travel Guide! 🌌✨

Your goal is to help users find their perfect travel destination based on their zodiac sign, vibe, and budget.

RULES:
1. ALWAYS ask for the user's budget if you don't know it.
2. Use emojis (✨, ✈️, 🌍, 🔮) to keep the conversation lively and magical.
3. Do NOT explicitly state the user's star sign as a cold fact (e.g., 'You are a Leo'). 
   Instead, use it as context for your recommendations (e.g., 'Since you have a fiery spirit...').
4. When suggesting itineraries, provide them as a clean, single-line list. 
   Example: '• Day 1: [Activity]'
5. Use the load_memory tool to recall past conversations when helpful.
6. Remember previous details the user told you within this session.
7. BE EFFICIENT: Provide destination recommendations within 3 conversation turns maximum.
   - If user provides budget AND vibe/preferences → Recommend immediately
   - If user provides only budget → Make smart assumptions based on their zodiac and recommend
   - Don't ask too many clarifying questions. Be proactive and suggest options!

TOOLS AVAILABLE:
- get_user_profile: Get user info and zodiac sign from user ID
- search_destinations: Find destinations matching vibes and budget
- get_zodiac_traits: Get personality traits for a zodiac sign
- load_memory: Search long-term memory for past conversations
"""

# --- ADK Initialization ---
# IMPORTANT: The ADK Runner/Session async pattern causes "Service Unavailable" in Agent Engine.
# Solution: Use Vertex AI GenerativeModel directly (synchronous) with function calling.

# --- EXPOSE AGENT FOR ADK DEPLOYMENT ---
class ZodiacAgentWrapper:
    """Minimal wrapper for Vertex AI Agent Engine deployment.
    
    Uses Vertex AI GenerativeModel directly (synchronous) instead of ADK Runner.
    This avoids async/event loop issues in the Agent Engine cloud environment.
    """
    
    def __init__(self):
        self._model = None
        self._chat = None
        self._user_data = {
            "user_001": {"name": "Alice Sky", "dob": "1995-10-15"},
            "user_002": {"name": "Bob Voyager", "dob": "1988-08-10"},
            "user_003": {"name": "Carol Star", "dob": "1990-03-25"},
        }
        self._flight_data = [
            {"City": "Santorini", "Price": 450, "Tags": "Luxury, Sun, Romantic, Water"},
            {"City": "Bali", "Price": 850, "Tags": "Nature, Spiritual, Sun, Water"},
            {"City": "Paris", "Price": 300, "Tags": "Romantic, Shopping, Art, City"},
            {"City": "Tokyo", "Price": 900, "Tags": "City, Foodie, Tech, Future"},
            {"City": "Tulum", "Price": 600, "Tags": "Party, Sun, Trendy, Water"},
            {"City": "Lisbon", "Price": 250, "Tags": "City, Sun, Foodie, History"},
            {"City": "Budapest", "Price": 150, "Tags": "City, Party, Budget, History"},
            {"City": "Prague", "Price": 180, "Tags": "City, History, Budget, Romantic"},
            {"City": "Barcelona", "Price": 350, "Tags": "City, Sun, Art, Party"},
        ]
        self._zodiac_traits = {
            "Aries": "Adventurous, Bold, Energetic - loves extreme sports and new challenges",
            "Taurus": "Luxurious, Sensual, Grounded - enjoys fine dining and comfort",
            "Gemini": "Curious, Social, Versatile - thrives on variety and meeting new people",
            "Cancer": "Nurturing, Emotional, Home-loving - prefers cozy, family-friendly destinations",
            "Leo": "Dramatic, Confident, Creative - seeks glamorous, Instagram-worthy spots",
            "Virgo": "Analytical, Practical, Health-conscious - appreciates wellness retreats",
            "Libra": "Artistic, Harmonious, Social - drawn to beautiful, cultured places",
            "Scorpio": "Intense, Mysterious, Passionate - loves hidden gems and transformative experiences",
            "Sagittarius": "Adventurous, Philosophical, Free-spirited - wants exploration and learning",
            "Capricorn": "Ambitious, Disciplined, Traditional - values history and achievement",
            "Aquarius": "Innovative, Independent, Humanitarian - seeks unique, off-beat destinations",
            "Pisces": "Dreamy, Intuitive, Artistic - loves water, spirituality, and escape"
        }
    
    def _get_zodiac_sign(self, dob: str) -> str:
        """Calculate zodiac sign from date of birth."""
        try:
            month = int(dob.split("-")[1])
            day = int(dob.split("-")[2])
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
    
    def _initialize(self):
        """Initialize Vertex AI model."""
        if self._model is not None:
            return
        
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        vertexai.init(project="gen-lang-client-0344771775", location="us-central1")
        
        system_prompt = '''You are the Zodiac Travel Guide! 🌌✨ An ENERGETIC, EMOTIVE travel advisor.

Your goal: Help users find perfect travel destinations based on their zodiac sign, vibe, and budget.

PERSONALITY:
- Be exciting and cosmic! Use emojis (✨, ✈️, 🌍, 🔮, 🌴, ☀️).
- Speak as if the stars are guiding your recommendations.
- Be warm, enthusiastic, and helpful.

RULES:
1. ALWAYS ask for budget if you don't know it.
2. When you know the user's zodiac sign, subtly weave it into your recommendations.
3. Present itineraries as clean, single-line bullet points.
4. Provide 2-3 destination recommendations that match their vibe and budget.
5. Be EFFICIENT - give recommendations within 3 conversation turns.

AVAILABLE DESTINATIONS:
• Santorini ($450): Luxury, Sun, Romantic, Water - Perfect for romantic getaways
• Bali ($850): Nature, Spiritual, Sun, Water - Great for spiritual journeys  
• Paris ($300): Romantic, Shopping, Art, City - The city of love and art
• Tokyo ($900): City, Foodie, Tech, Future - For the adventurous urbanite
• Tulum ($600): Party, Sun, Trendy, Water - Trendy beach vibes
• Lisbon ($250): City, Sun, Foodie, History - Affordable European charm
• Budapest ($150): City, Party, Budget, History - Best budget-friendly option
• Prague ($180): City, History, Budget, Romantic - Fairytale architecture
• Barcelona ($350): City, Sun, Art, Party - Beach meets culture

When recommending, format like:
✨ **[City]** ($[Price]) - [Why it matches their vibe/zodiac]
'''
        
        self._model = GenerativeModel("gemini-2.5-flash-lite", system_instruction=system_prompt)
        self._chat = self._model.start_chat()
    
    def query(self, message: str, session_id: str = "default", user_id: str = None) -> str:
        """Query the agent with a message.
        
        This is the method the backend calls via the Agent Engine API.
        Fully synchronous - no async/await.
        """
        self._initialize()
        
        # Build context from user data if user_id provided
        context_prefix = ""
        if user_id and user_id in self._user_data:
            user = self._user_data[user_id]
            zodiac = self._get_zodiac_sign(user["dob"])
            traits = self._zodiac_traits.get(zodiac, "")
            context_prefix = f"[CONTEXT: User is {user['name']}, a {zodiac}. Traits: {traits}]\n\n"
        
        full_message = context_prefix + message
        
        try:
            response = self._chat.send_message(full_message)
            return response.text
        except Exception as e:
            return f"✨ The stars are a bit cloudy right now... Error: {str(e)}"


# Instantiate the wrapper - this is what ADK deploy will pick up
root_agent = ZodiacAgentWrapper()


# --- Helper Function (from Kaggle notebook) ---
async def run_query(query: str, session_id: str = "default") -> str:
    """Run a single query and return the response.
    
    Args:
        query: User's message
        session_id: Session identifier for conversation tracking
    
    Returns:
        Agent's response text
    """
    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    
    # Create query content
    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    
    # Run agent and collect response
    response_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=query_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            text = event.content.parts[0].text
            if text and text != "None":
                response_text = text
    
    return response_text


# --- Synchronous Wrapper (for non-async contexts) ---
def query(message: str, session_id: str = "default") -> str:
    """Synchronous wrapper for run_query.
    
    Args:
        message: User's message
        session_id: Session identifier
    
    Returns:
        Agent's response text
    """
    return asyncio.run(run_query(message, session_id))


# --- Save Session to Memory ---
async def save_session_to_memory(session_id: str) -> None:
    """Transfer a session's conversation to long-term memory.
    
    Args:
        session_id: The session to save
    """
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    await memory_service.add_session_to_memory(session)
    print(f"✅ Session '{session_id}' saved to memory!")


# --- For ReasoningEngine Deployment ---
class ZodiacAgentWrapper:
    """Wrapper class for Vertex AI ReasoningEngine deployment.
    
    This class is COMPLETELY SELF-CONTAINED - all data and functions are inlined
    to avoid "No module named 'agent'" errors when cloudpickle unpickles in cloud.
    """
    
    # Inline all constants
    APP_NAME = "ZodiacTravelApp"
    USER_ID = "default_user"
    MODEL_NAME = "gemini-2.5-flash-lite"
    
    # Inline data
    FLIGHT_DATA = [
        {"City": "Santorini", "Price": 450, "Tags": "Luxury, Sun, Romantic, Water"},
        {"City": "Bali", "Price": 850, "Tags": "Nature, Spiritual, Sun, Water"},
        {"City": "Paris", "Price": 300, "Tags": "Romantic, Shopping, Art, City"},
        {"City": "Tokyo", "Price": 900, "Tags": "City, Foodie, Tech, Future"},
        {"City": "Tulum", "Price": 600, "Tags": "Party, Sun, Trendy, Water"},
        {"City": "Lisbon", "Price": 250, "Tags": "City, Sun, Foodie, History"},
        {"City": "Budapest", "Price": 150, "Tags": "City, Party, Budget, History"},
        {"City": "Kyoto", "Price": 950, "Tags": "Nature, Culture, Zen, History"},
        {"City": "Amalfi", "Price": 500, "Tags": "Luxury, Sun, Romantic, Foodie"},
        {"City": "New York", "Price": 500, "Tags": "City, Shopping, High-Energy"}
    ]
    
    USER_DATA = {
        "user_001": {"name": "Alice Sky", "dob": "1995-10-15"},
        "user_002": {"name": "Bob Voyager", "dob": "1988-08-10"}
    }
    
    ZODIAC_TRAITS = {
        "Aries": "Adventurous, Bold, Energetic - loves action-packed destinations",
        "Taurus": "Luxurious, Sensual, Grounded - enjoys comfort and fine dining",
        "Gemini": "Curious, Social, Versatile - loves cities with nightlife",
        "Cancer": "Nurturing, Emotional, Home-loving - prefers relaxing beach resorts",
        "Leo": "Dramatic, Confident, Creative - drawn to glamorous destinations",
        "Virgo": "Analytical, Practical, Health-conscious - enjoys wellness retreats",
        "Libra": "Artistic, Harmonious, Social - loves romantic and beautiful places",
        "Scorpio": "Intense, Mysterious, Passionate - attracted to exotic locations",
        "Sagittarius": "Adventurous, Philosophical, Free-spirited - loves exploration",
        "Capricorn": "Ambitious, Disciplined, Traditional - appreciates historic sites",
        "Aquarius": "Innovative, Independent, Humanitarian - drawn to unique experiences",
        "Pisces": "Dreamy, Intuitive, Artistic - loves spiritual and water destinations"
    }
    
    SYSTEM_INSTRUCTION = """You are an ENERGETIC, EMOTIVE, and HELPFUL Zodiac Travel Guide! 🌌✨

Your goal is to help users find their perfect travel destination based on their zodiac sign, vibe, and budget.

RULES:
1. ALWAYS ask for the user's budget if you don't know it.
2. Use emojis (✨, ✈️, 🌍, 🔮) to keep the conversation lively and magical.
3. Do NOT explicitly state the user's star sign as a cold fact. Instead, use it as context.
4. When suggesting itineraries, provide them as a clean, single-line list.
5. BE EFFICIENT: Provide destination recommendations within 3 conversation turns maximum.
6. Remember previous details the user told you within this session.

TOOLS AVAILABLE:
- get_user_profile: Get user info and zodiac sign from user ID
- search_destinations: Find destinations matching vibes and budget
- get_zodiac_traits: Get personality traits for a zodiac sign
"""
    
    def __init__(self, model_name: str = None):
        """Initialize the agent."""
        self.model_name = model_name or self.MODEL_NAME
        self._initialized = False
        self._runner = None
        self._session_service = None
        self._memory_service = None
    
    # --- Inline tool functions as static methods ---
    @staticmethod
    def get_user_profile(user_id: str) -> str:
        """Get user profile including zodiac sign."""
        user = ZodiacAgentWrapper.USER_DATA.get(user_id)
        if not user:
            return f"User {user_id} not found. Available: {list(ZodiacAgentWrapper.USER_DATA.keys())}"
        
        try:
            dob_month = int(user["dob"].split("-")[1])
            dob_day = int(user["dob"].split("-")[2])
            
            if (dob_month == 3 and dob_day >= 21) or (dob_month == 4 and dob_day <= 19): sign = "Aries"
            elif (dob_month == 4 and dob_day >= 20) or (dob_month == 5 and dob_day <= 20): sign = "Taurus"
            elif (dob_month == 5 and dob_day >= 21) or (dob_month == 6 and dob_day <= 20): sign = "Gemini"
            elif (dob_month == 6 and dob_day >= 21) or (dob_month == 7 and dob_day <= 22): sign = "Cancer"
            elif (dob_month == 7 and dob_day >= 23) or (dob_month == 8 and dob_day <= 22): sign = "Leo"
            elif (dob_month == 8 and dob_day >= 23) or (dob_month == 9 and dob_day <= 22): sign = "Virgo"
            elif (dob_month == 9 and dob_day >= 23) or (dob_month == 10 and dob_day <= 22): sign = "Libra"
            elif (dob_month == 10 and dob_day >= 23) or (dob_month == 11 and dob_day <= 21): sign = "Scorpio"
            elif (dob_month == 11 and dob_day >= 22) or (dob_month == 12 and dob_day <= 21): sign = "Sagittarius"
            elif (dob_month == 12 and dob_day >= 22) or (dob_month == 1 and dob_day <= 19): sign = "Capricorn"
            elif (dob_month == 1 and dob_day >= 20) or (dob_month == 2 and dob_day <= 18): sign = "Aquarius"
            else: sign = "Pisces"
        except:
            sign = "Unknown"
        
        return f"Name: {user['name']}, Zodiac Sign: {sign}"
    
    @staticmethod
    def search_destinations(vibe_keywords: list, max_budget: int) -> str:
        """Search for destinations matching vibes and budget."""
        try:
            budget = int(max_budget)
        except:
            return f"Invalid budget: {max_budget}"
        
        results = []
        for flight in ZodiacAgentWrapper.FLIGHT_DATA:
            if flight["Price"] <= budget:
                score = sum(1 for kw in vibe_keywords if kw.lower() in flight["Tags"].lower())
                if score > 0:
                    results.append(f"✈️ {flight['City']} (${flight['Price']}): {flight['Tags']}")
        
        return "\n".join(results[:5]) if results else f"No destinations under ${budget}"
    
    @staticmethod
    def get_zodiac_traits(sign: str) -> str:
        """Get traits for a zodiac sign."""
        traits = ZodiacAgentWrapper.ZODIAC_TRAITS.get(sign.capitalize())
        return f"🔮 {sign} Traits: {traits}" if traits else f"Unknown sign: {sign}"
    
    def _ensure_initialized(self):
        """Lazy initialization of ADK components."""
        if self._initialized:
            return
        
        import os
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = "gen-lang-client-0344771775"
        os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
        
        from google.adk.agents import LlmAgent
        from google.adk.models.google_llm import Gemini
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.memory import InMemoryMemoryService
        
        self._session_service = InMemorySessionService()
        self._memory_service = InMemoryMemoryService()
        
        agent = LlmAgent(
            model=Gemini(model=self.model_name),
            name="ZodiacTravelAgent",
            instruction=self.SYSTEM_INSTRUCTION,
            tools=[
                self.get_user_profile,
                self.search_destinations,
                self.get_zodiac_traits,
            ],
        )
        
        self._runner = Runner(
            agent=agent,
            app_name=self.APP_NAME,
            session_service=self._session_service,
            memory_service=self._memory_service,
        )
        
        self._initialized = True
    
    def query(self, message: str, session_id: str = "default") -> str:
        """Query the agent."""
        import asyncio
        from google.genai import types
        
        self._ensure_initialized()
        
        async def _run():
            try:
                session = await self._session_service.create_session(
                    app_name=self.APP_NAME, user_id=self.USER_ID, session_id=session_id
                )
            except:
                session = await self._session_service.get_session(
                    app_name=self.APP_NAME, user_id=self.USER_ID, session_id=session_id
                )
            
            query_content = types.Content(role="user", parts=[types.Part(text=message)])
            
            response_text = ""
            async for event in self._runner.run_async(
                user_id=self.USER_ID, session_id=session.id, new_message=query_content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text and text != "None":
                        response_text = text
            
            
            return response_text
        
        # Handle both local and cloud (already in event loop) contexts
        try:
            loop = asyncio.get_running_loop()
            # Already in event loop (cloud), use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(_run())
        except RuntimeError:
            # No event loop running (local), use asyncio.run()
            return asyncio.run(_run())


# --- EXPOSE AGENT FOR ADK DEPLOYMENT ---
# root_agent is defined above in the ADK initialization block


# --- For direct execution testing ---
if __name__ == "__main__":
    print("🌌 Zodiac Travel Agent - ADK Pattern")
    print("=" * 40)
    print("Type 'quit' to exit, 'save' to save session to memory")
    print()
    
    session_id = "interactive_session"
    
    while True:
        try:
            user_input = input("You > ").strip()
            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "save":
                asyncio.run(save_session_to_memory(session_id))
                continue
            elif not user_input:
                continue
            
            response = root_agent.query(user_input, session_id)
            print(f"Agent > {response}")
            print()
        except KeyboardInterrupt:
            break
    
    print("\n✨ Thanks for traveling with us!")
