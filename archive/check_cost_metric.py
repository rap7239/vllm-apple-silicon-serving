import json
import urllib.request
import urllib.parse

query = "run_bench_cost_per_1k_tokens_usd"
url = "http://localhost:9090/api/v1/query?" + urllib.parse.urlencode({"query": query})

with urllib.request.urlopen(url) as resp:
    data = json.load(resp)

for r in data["data"]["result"]:
    print(r["metric"], "=", r["value"][1])
