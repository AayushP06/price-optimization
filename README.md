## Dynamic Price Optimization (Flask + HTML)

Interactive price optimization tool using Quick Sort and Binary Search. The Flask backend serves the single-page site and exposes REST endpoints; the frontend includes a form to input product details and view the optimal price, margin, and expected profit.

### Features
- Input product name, quality (budget/standard/premium), cost price, fixed costs
- Optional: supply competitor prices or let backend generate realistic samples
- Computes optimal selling price with profit margin and expected profit
- Clean UI, reduced text with “Read more” toggles, Chart.js example

## Project Structure
```
flask-api-backend.py           
price-optimization-website.html
price-optimization-backend.py  
requirements.txt               
.gitignore                     
README.md                      
```

## Requirements
- Python 3.11 recommended
- Windows PowerShell (or any shell)

## Setup (Windows PowerShell)
```powershell
cd "C:\Users\aayus\Desktop\DS Project"

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```



## Run Locally
```powershell
.\.venv\Scripts\Activate.ps1
python flask-api-backend.py
```
Open `http://localhost:5000`

### Using the App
- Fill the form (product, quality, cost, fixed costs). Optional: competitor prices (comma-separated) or number of competitors.
- Click “Generate Optimal Price”. Results render under the form.

## API Endpoints
- POST `/api/optimize`
  - Request JSON (fields):
    ```json
    {
      "product_name": "Wireless Earbuds",
      "quality": "standard",
      "cost_price": 1000,
      "fixed_costs": 50,
      "competitor_prices": [1299, 1549, 1799],
      "num_competitors": 50,
      "min_margin": 15,
      "max_margin": 35
    }
    ```
  - Response JSON (excerpt):
    ```json
    {
      "success": true,
      "product": { "name": "Wireless Earbuds", "quality": "standard" },
      "data": {
        "optimal_price": { "price": 1599.0, "margin": 26.0, "expected_profit": 12345.0 },
        "all_prices": [ ... ],
        "competitor_stats": { ... },
        "performance": { "sort_time": 0.0012, "search_time": 0.0001 }
      }
    }
    ```

## Deploy (Render quick start)
1. Add Gunicorn to `requirements.txt`:
   ```
   gunicorn
   ```
2. Create `Procfile` in project root:
   ```
   web: gunicorn flask-api-backend:app
   ```
3. Push to GitHub, then on Render create a new Web Service from the repo.
4. Environment: Python 3.11. Render installs from `requirements.txt` and uses the `Procfile`.
5. Open your Render URL (root `/` serves the site; API under `/api/...`).


