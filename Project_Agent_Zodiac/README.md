# 🌌 Zodiac Travel Agent

A personalized, AI-powered travel recommendation agent that suggests destinations based on your **Zodiac sign**, vibe, and budget. Built with **Google Cloud Vertex AI Agent Engine**, **FastAPI**, and **React**.

[![Demo - Live](https://img.shields.io/badge/Demo-Live-brightgreen)](https://Chugh-Gourav.github.io/Project_Agent_Zodiac/)
![Google Cloud](https://img.shields.io/badge/Cloud-Google%20Cloud-4285F4)
![Vertex AI](https://img.shields.io/badge/AI-Vertex%20AI-FF6F00)
![Function Calling](https://img.shields.io/badge/Gemini-Function%20Calling-blueviolet)

## ✨ Features

- **🔮 Zodiac Personalization**: Recommendations tailored to your astrological personality traits
- **🛠️ Function Calling**: Real tool invocations (`search_destinations`, `get_user_profile`) for dynamic data retrieval
- **💬 Conversational AI**: Natural language interaction powered by Gemini 2.5 Flash
- **🧠 Session Memory**: Remembers your preferences within a conversation
- **☁️ Cloud-Native**: Deployed on Google Cloud (Agent Engine + Cloud Run)
- **🎨 Cosmic UI**: Beautiful, immersive React frontend with animations

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌───────────────────────────────────────┐
│                 │     │                 │     │       Vertex AI Agent Engine          │
│  React Frontend │────▶│  FastAPI        │────▶│  ┌─────────────────────────────────┐  │
│  (GitHub Pages) │     │  (Cloud Run)    │     │  │       ZodiacTravelAgent         │  │
│                 │◀────│                 │◀────│  │                                 │  │
└─────────────────┘     └─────────────────┘     │  │  ┌──────────────────────────┐   │  │
                                                │  │  │    Function Calling      │   │  │
                                                │  │  │  • search_destinations   │   │  │
                                                │  │  │  • get_user_profile      │   │  │
                                                │  │  └──────────────────────────┘   │  │
                                                │  │              │                  │  │
                                                │  │              ▼                  │  │
                                                │  │  ┌──────────────────────────┐   │  │
                                                │  │  │    Gemini 2.5 Flash      │   │  │
                                                │  │  │  (with system prompt)    │   │  │
                                                │  │  └──────────────────────────┘   │  │
                                                │  └─────────────────────────────────┘  │
                                                └───────────────────────────────────────┘
```

### 🔧 Agent Tools (Function Calling)

| Tool | Parameters | Description |
|------|------------|-------------|
| `search_destinations` | `max_budget`, `vibes[]` | Searches destinations within budget, filters by vibe tags |
| `get_user_profile` | `user_id` | Returns user's name, zodiac sign, and personality traits |

## 📁 Project Structure

```
zodiac-travel-agent/
├── frontend/              # React + Vite + Tailwind CSS
│   ├── src/App.jsx        # Main chat component
│   └── .env               # Backend URL configuration
├── backend/               # FastAPI proxy service
│   ├── app.py             # API endpoints + Agent Engine integration
│   └── requirements.txt   # Python dependencies
└── agent-source/          # Agent code for Vertex AI
    ├── agent.py           # ADK pattern implementation (LlmAgent, Runner)
    ├── deploy_sdk.py      # SDK deployment script (ReasoningEngine.create)
    ├── run_agent.py       # Local testing script
    └── requirements.txt   # Agent dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Google Cloud Project with Vertex AI API enabled
- `gcloud` CLI installed and authenticated

### 1. Deploy the Agent (Vertex AI)

```bash
cd agent-source

# Install dependencies
pip install -r requirements.txt

# Deploy using SDK (recommended)
python deploy_sdk.py
# Note the Agent ID printed at the end
```

### 2. Run the Backend (Cloud Run or Local)

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Update AGENT_ID in app.py with your deployed agent ID

# Run locally
uvicorn app:app --reload --port 8000

# Or deploy to Cloud Run
gcloud run deploy zodiac-backend --source . --region us-central1 --allow-unauthenticated
```

### 3. Run the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env file with backend URL
echo "VITE_API_URL=https://your-backend-url.run.app" > .env

# Run locally
npm run dev

# Or deploy to GitHub Pages
npm run deploy
```

## 🔑 Key Components

### Agent (`agent-source/agent.py`)

Uses the **ADK pattern** with:
- `LlmAgent`: Agent definition with tools and instructions
- `Runner`: Execution handler with session management
- `InMemorySessionService`: Conversation memory
- `InMemoryMemoryService`: Long-term knowledge storage

### SDK Deployment (`agent-source/deploy_sdk.py`)

Direct deployment using `ReasoningEngine.create()` for explicit control over:
- `query(input)` method signature
- User context handling
- Zodiac trait integration

### Backend (`backend/app.py`)

FastAPI service that:
- Adds system context to user messages
- Calls Agent Engine via REST API
- Falls back to direct Gemini if needed
- Handles CORS for frontend

## 🌐 Live Demo

- **Frontend**: [https://Chugh-Gourav.github.io/Project_Agent_Zodiac/](https://Chugh-Gourav.github.io/Project_Agent_Zodiac/)
- **Backend**: Cloud Run (zodiac-backend)
- **Agent Engine ID**: `5104826879089573888`

## 📊 Available Destinations

| City | Price | Tags |
|------|-------|------|
| Budapest | $150 | City, Party, Budget, History |
| Prague | $180 | City, History, Budget, Romantic |
| Lisbon | $250 | City, Sun, Foodie, History |
| Paris | $300 | Romantic, Shopping, Art, City |
| Barcelona | $350 | City, Sun, Art, Party |
| Santorini | $450 | Luxury, Sun, Romantic, Water |
| Tulum | $600 | Party, Sun, Trendy, Water |
| Bali | $850 | Nature, Spiritual, Sun, Water |
| Tokyo | $900 | City, Foodie, Tech, Future |

## 🔮 Zodiac Traits

The agent uses zodiac personality traits to personalize recommendations:

| Sign | Traits | Recommended Vibes |
|------|--------|-------------------|
| Aries | Bold, Energetic | Adventure, Action |
| Taurus | Luxurious, Grounded | Comfort, Fine Dining |
| Gemini | Social, Versatile | Nightlife, City Exploration |
| Leo | Confident, Creative | Glamorous, Instagram-worthy |
| Pisces | Dreamy, Intuitive | Spiritual, Water destinations |

## 🛠️ Tech Stack

- **Frontend**: React, Vite, Tailwind CSS, Framer Motion
- **Backend**: FastAPI, Uvicorn, Python
- **AI**: Google Vertex AI, Gemini 2.5 Flash, Agent Engine
- **Cloud**: Google Cloud Run, GitHub Pages

## 📝 Development Notes

### Local Testing

```bash
cd agent-source
python run_agent.py
# Enter queries interactively
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | GCP Project ID |
| `GOOGLE_CLOUD_LOCATION` | Region (us-central1) |
| `VITE_API_URL` | Backend URL for frontend |

## 📜 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- Built during Google's 5-Day AI Agent Course on Kaggle
- Inspired by zodiac-based personalization patterns
- Powered by Google Cloud Vertex AI
