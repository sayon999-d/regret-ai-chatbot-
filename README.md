# Regret AI

An AI-powered decision-making assistant that helps minimize regret by analyzing life decisions through counterfactual simulation and multi-agent reasoning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Counterfactual Simulation**: Explore alternative outcomes for your decisions
- **Multi-Agent Regret Council**: Aggregates perspectives from Risk-Averse, Optimist, Rational, and Long-Term agents
- **Emotion-Aware Filtering**: Tailors recommendations based on your emotional state
- **Semantic Memory**: Recalls similar past decisions using embeddings
- **Real-time Dashboard**: Beautiful web interface with live statistics and charts
- **Ollama Integration**: Uses local LLMs for intelligent outcome prediction

## Dashboard Preview

The dashboard includes:
- Decision form with emotion selection
- Regret trend line chart (1-10 scale)
- Action weights visualization
- Decision timeline
- Real-time health monitoring

## Quick Start

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) (optional, for AI-powered predictions)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sayon999-d/regret-ai-chatbot-.git
   cd regret-ai-chatbot-
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Start Ollama** (optional, for AI features)
   ```bash
   ollama serve
   ollama pull mistral  # or any other model
   ```

5. **Run the application**
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Open the dashboard**
   - Local: http://localhost:8000
   - Network: http://YOUR_IP:8000

## Mobile Access

To access from your mobile device:

1. Ensure your phone and computer are on the same WiFi
2. Find your computer's IP address:
   ```bash
   # macOS
   ipconfig getifaddr en0
   
   # Linux
   hostname -I
   
   # Windows
   ipconfig
   ```
3. Open `http://YOUR_IP:8000` on your mobile browser

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/health` | System health check |
| POST | `/decide` | Get decision recommendation |
| POST | `/outcome` | Submit decision outcome |
| POST | `/chat` | Conversational interface |

### Dashboard Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/dashboard/domains` | Regret by domain |
| GET | `/dashboard/actions` | Action weights |
| GET | `/dashboard/timeline` | Decision timeline |

### Example: Make a Decision

```bash
curl -X POST "http://localhost:8000/decide" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "context": "Should I switch jobs for higher pay but less stability?",
    "emotion": "anxiety"
  }'
```

**Response:**
```json
{
  "decision_id": "uuid-here",
  "user_id": "user123",
  "action": "Reflect before acting",
  "domain": "personal",
  "regret": 4.25,
  "timestamp": 1704369600.0
}
```

### Emotion Options

- `neutral` - Default, considers all domains
- `anxiety` - Focuses on health, life, personal decisions
- `sadness` - Focuses on health, relationships
- `anger` - Focuses on personal, relationships
- `confidence` - Focuses on career, finance, learning

## Architecture

```
+-----------------------------------------------------------+
|                    Regret AI System                        |
+-----------------------------------------------------------+
|  +--------------+  +--------------+  +-----------------+  |
|  |   Decision   |  | Counterfact  |  |  Regret Council |  |
|  |   Engine     |--|  Simulator   |--|  (Multi-Agent)  |  |
|  +--------------+  +--------------+  +-----------------+  |
|         |                |                   |            |
|         v                v                   v            |
|  +--------------+  +--------------+  +-----------------+  |
|  |   Regret     |  |   Regret     |  |     Ollama      |  |
|  |   Policy     |  |   Memory     |  |    Gateway      |  |
|  +--------------+  +--------------+  +-----------------+  |
+-----------------------------------------------------------+
```

## Decision Domains

| Domain | Example Actions |
|--------|-----------------|
| Career | Apply for job, Ask for promotion, Start business |
| Learning | Learn new skill, Online course, New language |
| Finance | Save money, Invest in stocks, Pay off debt |
| Health | Exercise, Improve diet, Mental health support |
| Relationships | Repair relationship, Family time |
| Life | Move to new city, Major life changes |
| Personal | Reflect, Focus on discipline |
| Digital | Reduce social media, Build online presence |

## Configuration

Key configuration variables in `main.py`:

```python
ALPHA = 0.6        # Weight for initial regret
BETA = 0.4         # Weight for outcome feedback
OLLAMA_URL = "http://localhost:11434"  # Ollama server URL
```

## Requirements

```
fastapi
uvicorn
httpx
numpy
pydantic
sentence-transformers
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [Chart.js](https://www.chartjs.org/) - Beautiful charts

---

Made by [Sayon Manna](https://github.com/sayon999-d)
