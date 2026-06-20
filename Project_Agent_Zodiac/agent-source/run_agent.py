"""
Zodiac Travel Agent - Local Testing Script

This script provides an interactive way to test the Zodiac Travel Agent
using the ADK pattern (LlmAgent + Runner).

Usage:
    python run_agent.py
"""

import asyncio
from agent import (
    runner,
    session_service,
    memory_service,
    APP_NAME,
    USER_ID,
    run_query,
    save_session_to_memory,
)
from google.genai import types


async def interactive_session():
    """Run an interactive session with the Zodiac Travel Agent."""
    print()
    print("🌌✨ Welcome to the Zodiac Travel Agent! ✨🌌")
    print("=" * 50)
    print()
    print("I'll help you find your perfect travel destination")
    print("based on your zodiac sign, vibe, and budget!")
    print()
    print("Commands:")
    print("  'quit' or 'exit' - End the session")
    print("  'save' - Save this session to long-term memory")
    print("  'new' - Start a new session")
    print("  'memory <query>' - Search long-term memory")
    print()
    print("-" * 50)
    
    session_id = "session_001"
    session_count = 1
    
    while True:
        try:
            user_input = input("\n🧑 You > ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["quit", "exit"]:
                print("\n✨ Thanks for traveling with us! Safe journeys! ✨")
                break
            
            elif user_input.lower() == "save":
                await save_session_to_memory(session_id)
                continue
            
            elif user_input.lower() == "new":
                session_count += 1
                session_id = f"session_{session_count:03d}"
                print(f"\n🔄 Started new session: {session_id}")
                continue
            
            elif user_input.lower().startswith("memory "):
                query_text = user_input[7:].strip()
                if query_text:
                    search_response = await memory_service.search_memory(
                        app_name=APP_NAME,
                        user_id=USER_ID,
                        query=query_text
                    )
                    print(f"\n🔍 Memory Search Results for '{query_text}':")
                    if search_response.memories:
                        for mem in search_response.memories:
                            if mem.content and mem.content.parts:
                                text = mem.content.parts[0].text[:100]
                                print(f"  [{mem.author}]: {text}...")
                    else:
                        print("  No memories found.")
                continue
            
            # Run query
            print("\n🔮 Agent > ", end="", flush=True)
            response = await run_query(user_input, session_id)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n✨ Session interrupted. Goodbye! ✨")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def demo_session():
    """Run a demo session to showcase the agent's capabilities."""
    print()
    print("🎬 Running Demo Session...")
    print("=" * 50)
    
    demo_queries = [
        ("Hi! I'm looking to plan a trip. I'm user_001.", "demo_001"),
        ("I have a budget of $500 and want something romantic.", "demo_001"),
        ("Tell me more about Santorini!", "demo_001"),
    ]
    
    for query, session_id in demo_queries:
        print(f"\n🧑 User > {query}")
        response = await run_query(query, session_id)
        print(f"🔮 Agent > {response}")
        await asyncio.sleep(1)  # Small delay between queries
    
    # Save the demo session to memory
    await save_session_to_memory("demo_001")
    
    print()
    print("=" * 50)
    print("🎬 Demo Complete!")
    print()
    print("Now testing cross-session memory retrieval...")
    print()
    
    # Test memory retrieval in a NEW session
    print("🧑 User > What destinations did we discuss?")
    response = await run_query("What destinations did we discuss?", "demo_002")
    print(f"🔮 Agent > {response}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        asyncio.run(demo_session())
    else:
        asyncio.run(interactive_session())
