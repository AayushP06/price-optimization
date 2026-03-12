# Dynamic Price Optimization System

An e-commerce price optimization tool using **Quick Sort** and **Binary Search** algorithms. Input a product's cost price and competitor prices; get back the optimal selling price that maximizes expected profit.

---

## Project Structure

```
price-optimization-system/
│
├── backend/                   Flask REST API
│   ├── app/
│   │   ├── api/routes.py      All API endpoints
│   │   ├── core/config.py     App configuration
│   │   ├── models/product.py  SQLAlchemy model
│   │   ├── services/          DS algorithms + PriceOptimizerAPI
│   │   ├── database/db.py     SQLAlchemy instance
│   │   ├── __init__.py        App factory (create_app)
│   │   └── main.py            Entry point
│   ├── requirements.txt
│   └── .env
│
├── frontend/                  React + Vite UI (port 3000)
│   ├── src/
│   │   ├── components/        Navbar, Hero, PriceOptimizer, Results, ProfitChart, Footer
│   │   ├── App.jsx
│   │   ├── index.jsx
│   │   └── styles.css
│   ├── index.html
│   └── package.json
│
├── ml-model/                  Standalone DS demo (no Flask needed)
│   ├── training.py            Quick Sort + Binary Search + matplotlib charts
│   └── dataset.csv            Sample product data
│
├── docs/
│   └── architecture.md        System architecture diagram
│
├── tests/
│   └── test_api.py            pytest API tests
│
├── docker/
│   └── Dockerfile             Container config for the backend
│
├── price-optimization-website.html   HTML frontend (served at / by Flask)
├── stored_data.json                  JSON history of past runs
├── render.yaml                        Render.com deploy config
└── .gitignore
```

---

## Quick Start

### 1. Setup virtual environment (first time only)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

### 2. Run the backend

```powershell
.\.venv\Scripts\Activate.ps1
python backend/app/main.py
```

Opens at **http://localhost:5000** (also serves the HTML frontend at `/`).

### 3. Run the React frontend (optional, newer UI)

```powershell
cd frontend
npm install        # first time only
npm run dev
```

Opens at **http://localhost:3000** — automatically proxies `/api/*` to the Flask backend.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/optimize` | Run full price optimization |
| POST | `/api/generate-sample` | Generate sample competitor prices |
| POST | `/api/sort` | Sort prices with Quick Sort |
| POST | `/api/search` | Search prices with Binary Search |
| GET | `/api/health` | Health check |
| GET | `/api/history` | Optimization history (DB) |
| GET | `/api/stats` | Stats (total runs, avg profit) |
| GET | `/api/view-history` | History from JSON file |

### Example: Optimize

```bash
POST http://localhost:5000/api/optimize
Content-Type: application/json

{
  "cost_price": 1000,
  "fixed_costs": 50,
  "num_competitors": 50,
  "min_margin": 15,
  "max_margin": 35
}
```

Response:
```json
{
  "success": true,
  "optimal_price": { "price": 1265.0, "margin": 20.2, "expected_profit": 2150.5 },
  "competitor_stats": { "count": 50, "min": 1050.0, "max": 1950.0 },
  "performance": { "search_time": 0.0001 }
}
```

---

## Run Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/test_api.py -v
```

---

## Standalone DS Demo

Run the algorithms without Flask (generates matplotlib charts):

```powershell
python ml-model/training.py
```

---

## Deploy to Render

1. Push to GitHub
2. On Render, create a **Web Service** from the repo
3. Set:
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `python backend/app/main.py`
   - **Environment Variable**: `DATABASE_URL` → your PostgreSQL connection string

---

## License

MIT
