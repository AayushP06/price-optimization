# Price Optimizer

A React/Vite application backed by a FastAPI service. It recommends a selling price from product costs and competitor prices, and reports expected profit, estimated units sold, and whether XGBoost or the heuristic fallback produced the result.

## Local development

Use Python 3.11 and Node.js 20 or newer. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
Copy-Item .env.example .env
```

Set `DATABASE_URL`, `SECRET_KEY`, `ADMIN_TOKEN`, `ENV`, and optionally `ENABLE_DOCS` in `.env`. Run the schema migrations and start FastAPI:

```powershell
cd backend
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

In another terminal, start Vite:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. Vite proxies `/api` requests to FastAPI on port 8000. The API docs are at `http://localhost:8000/docs` in development. Docs are enabled by default outside production and can be controlled with `ENABLE_DOCS`.

## Database migrations

Alembic reads `DATABASE_URL` from the environment or `.env`.

```powershell
cd backend
alembic upgrade head
alembic revision --autogenerate -m "describe schema change"
```

For a database whose schema already exactly matches the migration head, mark it without running the migrations:

```powershell
alembic stamp head
```

For a database with the original Flask schema (the `optimizations` table before the metadata columns), mark the baseline and then apply the metadata migration:

```powershell
alembic stamp 20261102_0001
alembic upgrade head
```

Do not stamp a database at head unless its schema already contains the current columns. A new local SQLite database is created by `alembic upgrade head`.

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/optimize` | Recommend a price and save optimization history. |
| POST | `/api/generate-sample` | Generate sample competitor prices. |
| POST | `/api/sort` | Sort up to 100,000 numeric prices. |
| POST | `/api/search` | Binary search a sorted price list. |
| GET | `/api/health` | Check database connectivity. |
| GET | `/api/history` | List saved results; supports `limit` and `product_id`. |
| GET | `/api/stats` | Return optimization count and average profit. |
| GET | `/api/products` | Return product IDs and optimization counts. |
| GET | `/admin/status` | Return database status as JSON; requires `Authorization: Bearer $ADMIN_TOKEN`. |

`GET /api/optimize` endpoint help was removed; use `/docs` for the OpenAPI documentation. Validation errors use status 400 and the response shape `{"success": false, "error": "Validation failed", "fields": {…}}`. `quality` accepts `budget`, `standard`, or `premium` and is metadata only; it does not affect the recommendation.

## Production and Docker

Render builds the frontend, applies Alembic migrations, and serves the SPA and API from one Uvicorn process. Set private `DATABASE_URL`, `SECRET_KEY`, and `ADMIN_TOKEN` values in the deployment environment. `ENV=production` disables docs by default; set `ENABLE_DOCS=true` to enable them.

Build and run from the repository root:

```bash
docker build -f docker/Dockerfile -t price-optimizer .
docker run -p 5000:5000 -e DATABASE_URL=sqlite:///./instance/local.db -e ADMIN_TOKEN=change-me price-optimizer
```

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -v
cd frontend
npm test
```
