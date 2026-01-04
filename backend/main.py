import uuid
import time
import random
import logging
import httpx
import numpy as np
from typing import Dict, List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ACTIONS

ACTIONS = {
    # CAREER
    "Apply for a new job": "career",
    "Switch current job": "career",
    "Ask for a promotion": "career",
    "Negotiate salary": "career",
    "Start a business": "career",
    "Work as a freelancer": "career",

    # LEARNING
    "Learn a new technical skill": "learning",
    "Pursue higher education": "learning",
    "Take an online course": "learning",
    "Learn a new language": "learning",

    # FINANCE
    "Save money": "finance",
    "Invest in stocks": "finance",
    "Invest in cryptocurrency": "finance",
    "Pay off debt": "finance",

    # HEALTH
    "Start exercising regularly": "health",
    "Improve diet": "health",
    "Seek mental health support": "health",
    "Improve sleep routine": "health",

    # RELATIONSHIPS
    "Repair a relationship": "relationships",
    "End a relationship": "relationships",
    "Spend more time with family": "relationships",

    # LIFE
    "Move to a new city": "life",
    "Move to a new country": "life",
    "Delay the decision": "life",

    # PERSONAL
    "Reflect before acting": "personal",
    "Focus on discipline": "personal",
    "Explore personal interests": "personal",

    # DIGITAL
    "Reduce social media usage": "digital",
    "Build online presence": "digital",
}

# EMOTION FILTERS

EMOTION_FILTER = {
    "anxiety": ["health", "life", "personal"],
    "sadness": ["health", "relationships"],
    "anger": ["personal", "relationships"],
    "confidence": ["career", "finance", "learning"],
    "neutral": list(set(ACTIONS.values()))
}

ALPHA = 0.6
BETA = 0.4
OLLAMA_URL = "http://localhost:11434"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("regret-ai")

# OLLAMA GATEWAY

class OllamaGateway:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url
        self.available = False
        self.model = "mistral"
        self._check_availability()

    def _check_availability(self):
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                self.available = True
                models = response.json().get("models", [])
                if models:
                    self.model = models[0].get("name", "mistral")
                logger.info(f"Ollama connected. Using model: {self.model}")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self.available = False

    def generate(self, prompt: str, context: str = "") -> Optional[str]:
        if not self.available:
            return None
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "context": context
                },
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")
        return None

    def predict_outcome(self, context: str, action: str) -> float:
        prompt = f"""You are an AI that predicts outcomes of life decisions.

Context: {context}
Decision: {action}

Rate the likely outcome from -10 (worst) to +10 (best).
Consider long-term consequences, emotional impact, and practical factors.
RESPOND WITH ONLY A NUMBER between -10 and 10."""

        response = self.generate(prompt)
        if response:
            try:
                score = float(response.strip().split()[0])
                return max(-10, min(10, score))
            except (ValueError, IndexError):
                pass
        return random.uniform(-10, 10)


# CORE COMPONENTS

class DecisionEngine:
    def choose(self, weights: Dict[str, float]) -> str:
        total = sum(weights.values())
        r = random.uniform(0, total)
        upto = 0
        for action, w in weights.items():
            if upto + w >= r:
                return action
            upto += w
        return random.choice(list(weights.keys()))


class CounterfactualSimulator:
    def __init__(self, ollama: OllamaGateway):
        self.ollama = ollama

    def simulate(self, context: str, actions: List[str]) -> Dict:
        outcomes = {}
        for action in actions:
            if self.ollama.available:
                score = self.ollama.predict_outcome(context, action)
                confidence = 0.85
            else:
                score = random.uniform(-10, 10)
                confidence = random.uniform(0.4, 1.0)
            
            outcomes[action] = {
                "score": score,
                "confidence": confidence
            }
        return outcomes


class RegretCouncil:
    def __init__(self):
        self.agents = {
            "RiskAverse": 1.3,
            "Optimist": 0.7,
            "Rational": 1.0,
            "LongTerm": 1.5
        }

    def aggregate(self, regret: float) -> float:
        weighted_scores = [regret * w for w in self.agents.values()]
        return sum(weighted_scores) / len(weighted_scores)


class RegretPolicy:
    def __init__(self, actions: Dict[str, str]):
        self.weights = {a: 1.0 for a in actions}

    def update(self, action: str, regret: float):
        self.weights[action] *= 0.9 if regret > 0 else 1.05

    def penalize(self, action: str):
        self.weights[action] *= 0.8

    def get_top_actions(self, n: int = 5) -> List[tuple]:
        sorted_actions = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_actions[:n]


class RegretMemory:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.records = []

    def store(self, record: Dict):
        record["embedding"] = self.encoder.encode(record["context"]).tolist()
        self.records.append(record)

    def recall(self, context: str, k=3):
        if not self.records:
            return []
        q = self.encoder.encode(context)
        sims = [(np.dot(np.array(r["embedding"]), q), r) for r in self.records]
        sims.sort(reverse=True)
        return [{k: v for k, v in r.items() if k != "embedding"} for _, r in sims[:k]]


class RegretEvaluator:
    def __init__(self):
        self.domain_regret = {}
        self.timeline = []

    def log(self, domain: str, regret: float):
        self.domain_regret.setdefault(domain, []).append(regret)
        self.timeline.append({"time": time.time(), "regret": round(regret, 2)})

    def report(self):
        return {
            d: round(sum(v) / len(v), 2)
            for d, v in self.domain_regret.items()
        }

# FASTAPI MODELS

class DecisionRequest(BaseModel):
    user_id: str
    context: str
    emotion: str = "neutral"


class OutcomeFeedback(BaseModel):
    decision_id: str
    outcome: str
    reflection: str = ""


class ChatRequest(BaseModel):
    user_id: str
    message: str


# SYSTEM INIT

ollama = OllamaGateway()
engine = DecisionEngine()
simulator = CounterfactualSimulator(ollama)
council = RegretCouncil()
policy = RegretPolicy(ACTIONS)
memory = RegretMemory()
evaluator = RegretEvaluator()

DECISIONS = {}
USER_PROFILES = {}


# FASTAPI APP

app = FastAPI(title="Regret AI", description="AI-powered decision making with regret minimization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# UI DASHBOARD

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Regret AI Dashboard</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='15' fill='%23fff'/><text x='50' y='70' font-size='60' font-weight='bold' text-anchor='middle' fill='%23000'>R</text></svg>">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #000000;
            --bg-secondary: #0a0a0a;
            --bg-card: #111111;
            --accent: #ffffff;
            --accent-dim: #888888;
            --text-primary: #ffffff;
            --text-secondary: #666666;
            --border: rgba(255, 255, 255, 0.15);
            --border-light: rgba(255, 255, 255, 0.08);
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            background: var(--accent);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #000;
            font-weight: 700;
            font-size: 1.2rem;
        }
        
        .logo h1 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }
        
        .status-badges {
            display: flex;
            gap: 1rem;
        }
        
        .badge {
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid var(--border);
            background: var(--bg-secondary);
        }
        
        .badge.online { border-color: var(--accent); }
        .badge.offline { border-color: var(--text-secondary); opacity: 0.5; }
        .badge .dot { 
            width: 6px; 
            height: 6px; 
            border-radius: 50%; 
            background: currentColor; 
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1.5rem;
            border: 1px solid var(--border-light);
            transition: all 0.2s ease;
        }
        
        .stat-card:hover {
            border-color: var(--border);
        }
        
        .stat-card h3 {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }
        
        .stat-card .value {
            font-size: 2rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 1.5rem;
        }
        
        @media (max-width: 1200px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        
        /* Cards */
        .card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1.5rem;
            border: 1px solid var(--border-light);
            margin-bottom: 1rem;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-light);
        }
        
        .card-header h2 {
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Decision Form */
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-group label {
            display: block;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: inherit;
            transition: all 0.2s ease;
        }
        
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .form-group textarea {
            min-height: 100px;
            resize: vertical;
        }
        
        .btn {
            width: 100%;
            padding: 0.875rem;
            background: var(--accent);
            border: none;
            border-radius: 4px;
            color: #000;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn:hover {
            opacity: 0.9;
        }
        
        .btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
        
        .btn-secondary {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-primary);
        }
        
        .btn-secondary:hover {
            border-color: var(--accent);
        }
        
        /* Results */
        .result-card {
            background: var(--bg-secondary);
            border: 1px solid var(--accent);
            border-radius: 4px;
            padding: 1.25rem;
            margin-top: 1rem;
            animation: slideIn 0.2s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-card h4 {
            color: var(--text-primary);
            margin-bottom: 1rem;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-light);
        }
        
        .result-item:last-child { border-bottom: none; }
        .result-item .label { color: var(--text-secondary); font-size: 0.9rem; }
        .result-item .value { font-weight: 500; }
        
        /* Domain Chart */
        .chart-container {
            height: 280px;
            margin-top: 0.5rem;
        }
        
        /* Action List */
        .action-list {
            max-height: 350px;
            overflow-y: auto;
        }
        
        .action-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.625rem 0.75rem;
            background: var(--bg-secondary);
            border-radius: 4px;
            margin-bottom: 0.375rem;
            border: 1px solid var(--border-light);
            transition: all 0.15s ease;
        }
        
        .action-item:hover {
            border-color: var(--border);
        }
        
        .action-item .name {
            font-weight: 400;
            font-size: 0.9rem;
            flex: 1;
        }
        
        .action-item .weight {
            background: var(--accent);
            color: #000;
            padding: 0.2rem 0.6rem;
            border-radius: 3px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        /* Timeline */
        .timeline {
            max-height: 280px;
            overflow-y: auto;
        }
        
        .timeline-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.625rem 0;
            border-bottom: 1px solid var(--border-light);
            font-size: 0.9rem;
        }
        
        .timeline-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        
        .timeline-dot.low { background: var(--accent); }
        .timeline-dot.medium { background: var(--accent-dim); }
        .timeline-dot.high { background: var(--text-secondary); }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb { background: var(--text-secondary); border-radius: 2px; }
        
        /* Loading */
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(0,0,0,0.2);
            border-radius: 50%;
            border-top-color: #000;
            animation: spin 0.8s linear infinite;
            margin-right: 0.5rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .error-text { color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="logo">
                <div class="logo-icon">R</div>
                <h1>Regret AI</h1>
            </div>
            <div class="status-badges">
                <div class="badge" id="ollamaStatus">
                    <span class="dot"></span>
                    <span>Ollama</span>
                </div>
                <div class="badge online">
                    <span class="dot"></span>
                    <span>API</span>
                </div>
            </div>
        </header>

        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Decisions</h3>
                <div class="value" id="totalDecisions">0</div>
            </div>
            <div class="stat-card">
                <h3>Active Users</h3>
                <div class="value" id="activeUsers">0</div>
            </div>
            <div class="stat-card">
                <h3>Avg Regret</h3>
                <div class="value" id="avgRegret">0.00</div>
            </div>
            <div class="stat-card">
                <h3>Domains</h3>
                <div class="value" id="domainCount">0</div>
            </div>
        </div>

        <!-- Main Grid -->
        <div class="main-grid">
            <div class="left-column">
                <!-- Domain Chart -->
                <div class="card">
                    <div class="card-header">
                        <h2>Regret by Domain</h2>
                    </div>
                    <div class="chart-container">
                        <canvas id="domainChart"></canvas>
                    </div>
                </div>

                <!-- Timeline -->
                <div class="card">
                    <div class="card-header">
                        <h2>Timeline</h2>
                    </div>
                    <div class="timeline" id="timeline">
                        <p style="color: var(--text-secondary); text-align: center; padding: 1rem;">No decisions yet</p>
                    </div>
                </div>

                <!-- Top Actions -->
                <div class="card">
                    <div class="card-header">
                        <h2>Action Weights</h2>
                    </div>
                    <div class="action-list" id="actionList">
                        <p style="color: var(--text-secondary); text-align: center; padding: 1rem;">Loading...</p>
                    </div>
                </div>
            </div>

            <div class="right-column">
                <!-- Decision Form -->
                <div class="card">
                    <div class="card-header">
                        <h2>New Decision</h2>
                    </div>
                    <form id="decisionForm">
                        <div class="form-group">
                            <label>User ID</label>
                            <input type="text" id="userId" value="user_demo" required>
                        </div>
                        <div class="form-group">
                            <label>Context</label>
                            <textarea id="context" placeholder="Describe your situation..." required></textarea>
                        </div>
                        <div class="form-group">
                            <label>Emotion</label>
                            <select id="emotion">
                                <option value="neutral">Neutral</option>
                                <option value="anxiety">Anxious</option>
                                <option value="sadness">Sad</option>
                                <option value="anger">Angry</option>
                                <option value="confidence">Confident</option>
                            </select>
                        </div>
                        <button type="submit" class="btn" id="submitBtn">
                            Get Recommendation
                        </button>
                    </form>
                    <div id="result"></div>
                </div>

                <!-- Quick Links -->
                <div class="card">
                    <div class="card-header">
                        <h2>API</h2>
                    </div>
                    <a href="/docs" class="btn btn-secondary" style="display: block; text-align: center; text-decoration: none; margin-bottom: 0.5rem;">
                        Swagger Docs
                    </a>
                    <a href="/redoc" class="btn btn-secondary" style="display: block; text-align: center; text-decoration: none;">
                        ReDoc
                    </a>
                </div>
            </div>
        </div>
    </div>

    <script>
        let domainChart;
        
        document.addEventListener('DOMContentLoaded', () => {
            initChart();
            loadHealth();
            loadDomains();
            loadActions();
            loadTimeline();
            setInterval(loadHealth, 10000);
        });

        function initChart() {
            const ctx = document.getElementById('domainChart').getContext('2d');
            domainChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Average Regret',
                        data: [],
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.5,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        cubicInterpolationMode: 'monotone'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { 
                        legend: { display: false }
                    },
                    scales: {
                        y: { 
                            min: 1,
                            max: 10,
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { 
                                color: '#888',
                                stepSize: 1,
                                font: { size: 11 }
                            }
                        },
                        x: { 
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { 
                                color: '#888',
                                font: { size: 11 }
                            }
                        }
                    }
                }
            });
        }

        async function loadHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('totalDecisions').textContent = data.decisions;
                document.getElementById('activeUsers').textContent = data.users;
                
                const ollamaStatus = document.getElementById('ollamaStatus');
                ollamaStatus.className = data.ollama_available ? 'badge online' : 'badge offline';
            } catch (e) { console.error(e); }
        }

        async function loadDomains() {
            try {
                const res = await fetch('/dashboard/domains');
                const data = await res.json();
                const domains = Object.keys(data);
                const values = Object.values(data);
                
                document.getElementById('domainCount').textContent = domains.length;
                document.getElementById('avgRegret').textContent = values.length > 0 
                    ? (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2) 
                    : '0.00';
                
                domainChart.data.labels = domains.map(d => d.charAt(0).toUpperCase() + d.slice(1));
                domainChart.data.datasets[0].data = values.map(v => Math.min(10, Math.max(1, v)));
                domainChart.update();
            } catch (e) { console.error(e); }
        }

        async function loadActions() {
            try {
                const res = await fetch('/dashboard/actions');
                const data = await res.json();
                const actionList = document.getElementById('actionList');
                
                const sorted = Object.entries(data).sort((a, b) => b[1] - a[1]);
                
                if (sorted.length === 0) {
                    actionList.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 1rem;">No actions yet</p>';
                    return;
                }
                
                actionList.innerHTML = sorted.map(([action, weight]) => `
                    <div class="action-item">
                        <span class="name">${action}</span>
                        <span class="weight">${weight.toFixed(2)}</span>
                    </div>
                `).join('');
            } catch (e) { console.error(e); }
        }

        async function loadTimeline() {
            try {
                const res = await fetch('/dashboard/timeline');
                const data = await res.json();
                const timeline = document.getElementById('timeline');
                
                if (data.length === 0) {
                    timeline.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 1rem;">No decisions yet</p>';
                    return;
                }
                
                timeline.innerHTML = data.slice(-20).reverse().map(item => {
                    const level = item.regret < 2 ? 'low' : item.regret < 5 ? 'medium' : 'high';
                    const time = new Date(item.time * 1000).toLocaleTimeString();
                    return `
                        <div class="timeline-item">
                            <div class="timeline-dot ${level}"></div>
                            <span>${time}</span>
                            <span style="margin-left: auto; font-weight: 500;">${item.regret}</span>
                        </div>
                    `;
                }).join('');
            } catch (e) { console.error(e); }
        }

        document.getElementById('decisionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span>Processing...';
            
            try {
                const res = await fetch('/decide', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: document.getElementById('userId').value,
                        context: document.getElementById('context').value,
                        emotion: document.getElementById('emotion').value
                    })
                });
                
                const data = await res.json();
                
                document.getElementById('result').innerHTML = `
                    <div class="result-card">
                        <h4>Recommendation</h4>
                        <div class="result-item">
                            <span class="label">Suggested Action</span>
                            <span class="value">${data.action}</span>
                        </div>
                        <div class="result-item">
                            <span class="label">Domain</span>
                            <span class="value">${data.domain}</span>
                        </div>
                        <div class="result-item">
                            <span class="label">Estimated Regret</span>
                            <span class="value">${data.regret.toFixed(2)}</span>
                        </div>
                        <div class="result-item">
                            <span class="label">Decision ID</span>
                            <span class="value" style="font-size: 0.75rem;">${data.decision_id}</span>
                        </div>
                    </div>
                `;
                
                loadHealth();
                loadDomains();
                loadActions();
                loadTimeline();
                
            } catch (err) {
                document.getElementById('result').innerHTML = `
                    <div class="result-card" style="border-color: #666;">
                        <h4>Error</h4>
                        <p class="error-text">${err.message}</p>
                    </div>
                `;
            }
            
            btn.disabled = false;
            btn.innerHTML = 'Get Recommendation';
        });
    </script>
</body>
</html>
"""


# ROOT & UI ENDPOINTS

@app.get("/", response_class=HTMLResponse)
def root():
    return DASHBOARD_HTML


# CORE ENDPOINTS

@app.post("/decide")
def decide(req: DecisionRequest):
    allowed_domains = EMOTION_FILTER.get(req.emotion, EMOTION_FILTER["neutral"])
    candidate_actions = [
        a for a, d in ACTIONS.items() if d in allowed_domains
    ]

    weights = {a: policy.weights[a] for a in candidate_actions}
    action = engine.choose(weights)

    outcomes = simulator.simulate(req.context, candidate_actions)
    best = max(outcomes, key=lambda a: outcomes[a]["score"])
    regret = (outcomes[best]["score"] - outcomes[action]["score"]) * outcomes[action]["confidence"]
    total_regret = council.aggregate(regret)
    total_regret = max(1, min(10, total_regret))  # Clamp to 1-10 range

    decision_id = str(uuid.uuid4())
    domain = ACTIONS[action]

    record = {
        "decision_id": decision_id,
        "user_id": req.user_id,
        "context": req.context,
        "emotion": req.emotion,
        "action": action,
        "domain": domain,
        "regret": round(total_regret, 2),
        "timestamp": time.time()
    }

    memory.store(record.copy())
    evaluator.log(domain, total_regret)
    policy.update(action, total_regret)
    DECISIONS[decision_id] = record

    return record


@app.post("/outcome")
def outcome(feedback: OutcomeFeedback):
    d = DECISIONS.get(feedback.decision_id)
    if not d:
        return {"error": "Decision not found"}

    outcome_score = {"better": -5, "neutral": 0, "worse": 5}.get(feedback.outcome, 0)
    final_regret = ALPHA * d["regret"] + BETA * outcome_score

    d["final_regret"] = round(final_regret, 2)
    d["reflection"] = feedback.reflection
    policy.update(d["action"], final_regret)

    return {"status": "updated", "final_regret": d["final_regret"]}


@app.post("/chat")
def chat_with_ai(req: ChatRequest):
    if not ollama.available:
        return {"error": "Ollama not available", "response": None}
    
    prompt = f"""You are a helpful AI assistant focused on decision-making and reducing regret.
    
User: {req.message}

Provide thoughtful, empathetic advice. Focus on practical steps and long-term thinking."""
    
    response = ollama.generate(prompt)
    return {"response": response or "I'm having trouble generating a response right now."}


@app.get("/recall/{user_id}")
def recall_decisions(user_id: str, context: str = ""):
    if context:
        return {"similar": memory.recall(context)}
    
    user_decisions = [d for d in DECISIONS.values() if d["user_id"] == user_id]
    return {"decisions": user_decisions[-10:]}


# DASHBOARD ENDPOINTS

@app.get("/dashboard/domains")
def domain_dashboard():
    return evaluator.report()


@app.get("/dashboard/actions")
def action_dashboard():
    stats = {}
    for d in DECISIONS.values():
        stats.setdefault(d["action"], []).append(d["regret"])
    return {k: round(sum(v)/len(v), 2) for k, v in stats.items()}


@app.get("/dashboard/timeline")
def timeline():
    return evaluator.timeline


@app.get("/dashboard/policy")
def policy_weights():
    return policy.weights

# SYSTEM ENDPOINTS

@app.get("/health")
def health():
    return {
        "status": "ok",
        "decisions": len(DECISIONS),
        "users": len(set(d["user_id"] for d in DECISIONS.values())),
        "ollama_available": ollama.available,
        "ollama_model": ollama.model if ollama.available else None
    }


@app.get("/actions")
def list_actions():
    return {
        "actions": ACTIONS,
        "emotions": list(EMOTION_FILTER.keys())
    }


# RUN SERVER

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 