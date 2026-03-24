import urllib.request
import json
data = json.dumps({"participant_id": "test", "polar_level": 1, "category": "C", "subtype": 1, "modifiers": ["M1"]}).encode('utf-8')
req = urllib.request.Request("http://127.0.0.1:8000/api/session/start", data=data, headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    print(e)
    if hasattr(e, 'read'):
        print(e.read().decode('utf-8'))
