"""Send N identical streaming requests at the same moment; report decode TPOT per request."""
import asyncio
import json
import statistics
import sys
import time

import httpx

URL = "http://127.0.0.1:8000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer local-vllm-key"}
MODEL = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
MAX_TOKENS = 256


async def one(client, t0):
    payload = {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": 0, "ignore_eos": True,
               "messages": [{"role": "user", "content": "Explain how a CPU cache works, in detail."}],
               "stream": True, "stream_options": {"include_usage": True}}
    first, out = None, None
    async with client.stream("POST", URL, json=payload, headers=HEADERS, timeout=600) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                out = chunk["usage"]["completion_tokens"]
            if first is None and (chunk.get("choices") or [{}])[0].get("delta", {}).get("content"):
                first = time.perf_counter()
    end = time.perf_counter()
    return {"ttft_ms": (first - t0) * 1000, "tpot_ms": (end - first) * 1000 / (out - 1), "end": end}


async def main(n):
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None)) as client:
        t0 = time.perf_counter()
        rs = await asyncio.gather(*(one(client, t0) for _ in range(n)))
    wall = max(r["end"] for r in rs) - t0
    print(f"N={n}  TTFT ms: min {min(r['ttft_ms'] for r in rs):.0f}  max {max(r['ttft_ms'] for r in rs):.0f}")
    print(f"N={n}  TPOT ms per request: {sorted(round(r['tpot_ms'], 1) for r in rs)}")
    print(f"N={n}  median TPOT {statistics.median(r['tpot_ms'] for r in rs):.1f} ms, "
          f"aggregate {n * MAX_TOKENS / wall:.1f} tok/s")


asyncio.run(main(int(sys.argv[1])))