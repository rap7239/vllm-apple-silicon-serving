import json
import urllib.request

with urllib.request.urlopen("http://localhost:9090/api/v1/targets") as resp:
    data = json.load(resp)

for t in data["data"]["activeTargets"]:
    print(t["labels"]["job"], "->", t["health"])
