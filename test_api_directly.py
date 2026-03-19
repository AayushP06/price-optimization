import urllib.request
import json
req = urllib.request.Request(
    'http://localhost:5000/api/optimize', 
    data=b'{"cost_price": 1000, "num_competitors": 10}', 
    headers={'Content-Type': 'application/json'}
)
try:
    with urllib.request.urlopen(req) as response:
        data = response.read().decode()
        with open('output.json', 'w') as f:
            f.write(data)
except Exception as e:
    print(e)
