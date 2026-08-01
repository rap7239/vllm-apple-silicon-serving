# Learning and Portfolio Standard

## Learning preference

Start with a working, hands-on experiment. Introduce theory while inspecting,
changing, breaking, measuring, and repairing the system—not as a long
prerequisite lecture.

For every task, use this loop:

1. **Build:** run the smallest useful end-to-end example.
2. **Observe:** inspect logs, requests, responses, memory use, and latency.
3. **Explain:** connect each observation to the underlying mechanism from first
   principles.
4. **Change:** vary one input or configuration and predict the result.
5. **Break and repair:** trigger a realistic failure, diagnose it, and document
   the fix.
6. **Demonstrate:** package the result as a concise employer-facing walkthrough.

## First-principles standard

Explanations should reach the grassroots level without assuming terminology is
self-explanatory. For example, a vLLM task should connect the HTTP request to
tokenization, prefill, KV-cache allocation, iterative decoding, scheduling,
sampling, detokenization, and the HTTP response.

For each important concept, capture:

- What problem exists without it?
- What does the mechanism store, compute, or coordinate?
- What are its inputs and outputs?
- Which resource does it trade: memory, compute, latency, or throughput?
- What observable evidence shows that it is working?
- When would an engineer choose a different approach?

## Employer-ready evidence

Every completed task should leave behind:

- A reproducible README with setup and run commands.
- Checked-in scripts instead of command-history-only work.
- A small architecture or request-flow explanation.
- Captured benchmark results with hardware and configuration.
- At least one documented failure and root-cause analysis.
- A list of design decisions and tradeoffs.
- A two-to-five-minute demo script.
- Interview prompts and concise answers based on the implementation.
- Clear separation between personal implementation and third-party components.

## Demonstration narrative

Use this structure when presenting a task:

1. **Problem:** what was being built and why.
2. **Constraints:** hardware, memory, platform, model, and compatibility limits.
3. **Architecture:** how a request moves through the system.
4. **Live proof:** start the service and call the API.
5. **Internals:** connect logs and metrics to first-principles concepts.
6. **Engineering judgment:** explain failures, fixes, and tradeoffs.
7. **Next experiment:** identify a measurable improvement.

## Task 1 evidence already available

The local vLLM deployment demonstrates:

- Apple Metal/MLX model execution on an M4 Mac mini.
- A quantized Mistral-7B model served through an OpenAI-compatible API.
- Native paged-attention kernels and a 7.06 GB KV cache.
- Reproducible model and API test scripts.
- Diagnosis of an `xgrammar`/TVM-FFI native ABI crash.
- Diagnosis of an incorrect distributed-network interface selection.
- Compatibility pins and loopback networking fixes.

The next work on Task 1 should add measured latency/throughput experiments and
a polished short demo rather than treating successful startup as the end.

