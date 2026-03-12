# Architecture

## System Overview

```
price-optimization-system/
│
├── backend/              Python (Flask) API server
├── frontend/             React (Vite) UI
├── ml-model/             Standalone DS algorithm scripts
├── tests/                pytest API tests
└── docker/               Container config
```

## Backend Layer Diagram

```
frontend (React :3000)          or        browser (HTML at :5000/)
          │                                        │
          └────────── POST /api/optimize ──────────┘
                              │
                      Flask (main.py)
                              │
             ┌────────────────┼─────────────────┐
             ▼                ▼                 ▼
        api/routes.py   models/product.py   database/db.py
             │                                   │
             ▼                                   ▼
  services/price_optimizer.py              SQLite / PostgreSQL
    ├── quick_sort()                       (stored via SQLAlchemy)
    ├── binary_search()
    └── PriceOptimizerAPI.analyze()
```

## Data Flow for `/api/optimize`

1. **Request** — `POST /api/optimize` with `{ cost_price, competitor_prices, ... }`
2. **Routes** — validates input, builds `PriceOptimizerAPI`
3. **Service** — runs Quick Sort on competitor prices, tests 13 price points with Binary Search
4. **Response** — returns `{ optimal_price, all_prices, competitor_stats, performance }`
5. **Persistence** — result stored in `stored_data.json` and database `optimizations` table

## Algorithms

| Algorithm | Where Used | Complexity |
|-----------|-----------|------------|
| Quick Sort | Sort competitor prices | O(n log n) avg |
| Binary Search | Find optimal price range | O(log n) |

## How to Run

### Backend
```powershell
.\.venv\Scripts\Activate.ps1
python backend/app/main.py
# → http://localhost:5000
```

### Frontend
```powershell
cd frontend
npm run dev
# → http://localhost:3000
```

### Tests
```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/test_api.py -v
```

### Standalone ML Demo
```powershell
.\.venv\Scripts\Activate.ps1
python ml-model/training.py
```
