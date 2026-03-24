import urllib.request
import json
data = json.dumps({"polar_level": 2, "category": "G", "subtype": 3, "modifiers": ["M1"]}).encode('utf-8')
req = urllib.request.Request("http://127.0.0.1:8000/api/settings", data=data, headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    if hasattr(e, 'read'):
        err = json.loads(e.read().decode('utf-8'))
        print(err['error'])
