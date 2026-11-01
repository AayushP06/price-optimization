# Price Optimizer

A Flask and React application that recommends a selling price from product costs and competitor prices. Recommendations include an expected net margin, estimated units sold, and expected profit. The model reports whether it used XGBoost or the heuristic fallback.

## Project layout

```text
backend/
  app/api/                 Flask blueprints for optimize, utilities, history, and health/admin
  app/services/            Optimization service, pricing algorithms, and model inference
  app/models/product.py    SQLAlchemy optimization history model
  app/schemas.py           Request validation
  app/api/core/config.py   Environment based configuration
  app/main.py              Flask/Gunicorn application entry point
  requirements.txt         Pinned Python dependencies
frontend/
  src/api/                 HTTP client
  src/hooks/               Optimizer submission and state
  src/components/          ProductForm, ResultCard, MarketPosition, ProfitCurve
  src/styles.css           Design tokens and responsive layout
ml-model/
  train_xgb.py             Train the JSON XGBoost model
  xgb_model.json           Trained model
  feature_columns.json     Model feature contract
  dataset.csv              Source data
  cleaned_products.csv     Training data
  data_cleaning.py         Dataset preparation
  visualize_xgboost.py     Model visualization utility
tests/test_api.py          Flask API and optimizer tests
docker/Dockerfile          Multi-stage frontend and backend image
render.yaml                Render deployment configuration
.env.example               Local environment variable template
```

## Local development

Python 3.11 and Node.js 20 or newer are required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
Copy-Item .env.example .env
```

Set `DATABASE_URL`, `SECRET_KEY`, and `ADMIN_TOKEN` in `.env` for local use. Start Flask in one terminal:

```powershell
python backend/app/main.py
```

Start the React/Vite development server in another:

```powershell
cd frontend
npm ci
npm run dev
```

Vite serves the UI at `http://localhost:3000` and proxies `/api` requests to Flask on port 5000. In production, the frontend build is served by the same Flask/Gunicorn process as the API.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET, POST | `/api/optimize` | Read endpoint docs or create a price recommendation. |
| POST | `/api/generate-sample` | Generate sample competitor prices. |
| POST | `/api/sort` | Sort up to 5,000 numeric prices. |
| POST | `/api/search` | Binary search a sorted price list. |
| GET | `/api/health` | Database health status. |
| GET | `/api/history` | Database backed optimization history; supports `limit` and `product_id`. |
| GET | `/api/stats` | Optimization count and average profit. |
| GET | `/api/products` | Product IDs and optimization counts. |
| GET | `/admin/status` | HTML database status, protected by `Authorization: Bearer $ADMIN_TOKEN`. |

Example request:

```json
{
  "cost_price": 1000,
  "fixed_costs": 50,
  "min_markup_pct": 5,
  "max_markup_pct": 80,
  "competitor_prices": [1299, 1549, 1799],
  "available_units": 100,
  "category": "Electronics",
  "quality": "standard"
}
```

`min_markup_pct` and `max_markup_pct` constrain price as a percentage over cost. The result's `margin_pct` is net margin after fixed costs as a percentage of selling price. The previous request names `min_margin` and `max_margin`, and response names `margin` and `markup`, remain accepted/returned for compatibility. `quality` is validated and stored as metadata only; it does not change the recommendation.

The optimizer sorts competitor prices with Python's built-in sort. Quick Sort and Binary Search are available on their utility endpoints; they are not claimed as the optimization algorithm.

Optimization history is stored in SQLAlchemy. This project uses `db.create_all()` at startup, which creates missing tables but does not migrate existing ones. Existing databases need a one-time schema migration or table recreation before using the added metadata columns (`category`, `month`, `rating`, `ad_spend`, `method`, `product_name`, and `quality`).

## Train the model

```powershell
.\.venv\Scripts\Activate.ps1
python ml-model/train_xgb.py
```

Training writes `ml-model/xgb_model.json` and `ml-model/feature_columns.json`. Inference checks the feature contract and falls back to the heuristic optimizer when model files are missing or incompatible.

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -v
cd frontend
npm test
```

## Production and Docker

Render builds the frontend with `npm ci && npm run build`, then runs the Flask app using Gunicorn. The `DATABASE_URL` value is provided by the Render database. Set `ADMIN_TOKEN` and `SECRET_KEY` as private environment variables in the Render service.

Build and run the container from the repository root:

```bash
docker build -f docker/Dockerfile -t price-optimizer .
docker run -p 5000:5000 -e DATABASE_URL=sqlite:///local.db -e ADMIN_TOKEN=change-me price-optimizer
```
