"""Raw MLX batched decode: N identical prompts in one batch, no server."""
import sys

from mlx_lm import batch_generate, load

MODEL = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
n = int(sys.argv[1])
  
model, tokenizer = load(MODEL)
messages = [{"role": "user", "content": "Explain how a CPU cache works, in detail."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
batch_generate(model, tokenizer, [prompt] * n, max_tokens=256)  # warm-up, discarded
stats = batch_generate(model, tokenizer, [prompt] * n, max_tokens=256).stats
print(f"N={n}  {stats.generation_tokens} tokens, {stats.generation_tps:.1f} tok/s, "
      f"step time {1000 * n / stats.generation_tps:.1f} ms")