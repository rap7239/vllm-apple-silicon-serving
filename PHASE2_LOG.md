# Phase 2 Log — Kubernetes vLLM Deployment

Running narrative log of decisions, mistakes, and lessons learned while working
through Phase 2 of the LLM Performance Engineering Master Plan (11 numbered
tasks + retrospective, weeks 5-6). Same purpose as `PHASE1_LOG.md`: source
material for a "lessons learned" writeup, not just a changelog.

> **Standing instruction for Claude (any session, new or continued):** at the
> end of each hands-on work session in this repo, append a new dated entry to
> this file summarizing that session's narrative — what was attempted, what
> broke, root causes, fixes, and the lesson worth remembering. Do this before
> the session is considered closed, without waiting to be reminded. If this
> file is visible to you (attached, read, or in this repo), treat this
> instruction as active regardless of what chat session you're in. Keep
> entries technical/professional — personal reflections belong in the
> local-only `NOTES_PERSONAL.md`, not here.

## Environment-adaptation decision

**Date:** 2026-08-13

No GPU node exists for this project (M4 Mac mini, no cluster) — same shape
of gap as Phase 1's item #5 (no DCGM/CUDA on Apple Silicon). Chosen path,
the user's explicit call: local Kubernetes via `kind`, no GPU scheduling.
Deploy real vLLM (CPU backend, not `vllm-metal`) as a regular pod with
`nvidia.com/gpu` requests/tolerations *dropped*, not faked or stubbed —
everything else (Helm chart, Service, Ingress, ServiceMonitor→Grafana
wiring, Job-based harness run, ArgoCD sync) stays real. The gap is stated
explicitly in-repo (Dockerfile/deployment comments), matching the honesty
pattern set by Phase 1's powermetrics-for-DCGM substitution rather than
glossing over it.

Tooling check: `kubectl` and `docker` were already installed; `kind`,
`minikube`, `helm` were not — installed via `brew` as task #1 began, user
driving the install personally per the project's working-style convention
(see `PHASE1_LOG.md`'s working-style note).

## Entry: Task #1 — Deployment YAML, seven bugs deep

**Date:** 2026-08-13/14

Stood up a `kind` cluster (`vllm-local`, control-plane + worker,
`k8s/kind-config.yaml`). Model choice: `Qwen/Qwen2.5-0.5B-Instruct`, picked
over `opt-125m` for real instruction-tuned output worth actually looking at.
Target: a real, working pod (`k8s/deployment.yaml` + `k8s/Dockerfile`),
confirmed `1/1 Running`, 0 restarts, `ready=true`.

Getting there took seven distinct real bugs, each one only surfacing once
the prior one cleared — worth recording in sequence since the shape of the
chain (each fix unmasking the next failure) is itself the lesson, not just
any individual fix:

1. **`pip install torch==2.11.0+cpu` failed.** That build only exists on
   PyTorch's own wheel index, not PyPI. Fixed with `--extra-index-url
   https://download.pytorch.org/whl/cpu`.
2. **Model download failed.** `HF_HUB_OFFLINE=1` had been set *before* the
   download `RUN` step in the Dockerfile instead of after — my own ordering
   mistake, not a vLLM/HF issue. Fixed by moving the offline flag to after
   the download step.
3. **Pod crash-looped, no useful error in the pod logs.** Kubernetes pods
   default to a 64MB `/dev/shm`. Confirmed live via a throwaway debug pod
   rather than assumed from docs. vLLM's CPU-backend multiprocess workers
   need much more than that for tensor IPC, so the child process died via a
   silent `SIGBUS` — no Python traceback at all, which is what made this one
   genuinely hard to diagnose from logs alone. Fixed with a 2Gi
   memory-backed `emptyDir` mounted at `/dev/shm`.
4. **Same crash persisted after the shm fix.** `kind load docker-image` had
   been skipped after a rebuild, so the node was still running a stale
   image under the same tag. Caught by directly comparing image IDs
   (`docker images` on the host vs. `crictl images` run inside the kind
   node via `docker exec`) — a real image-ID mismatch, not a guess.
5. **`libnuma.so.1: cannot open shared object file`.** `python:3.12-slim`
   doesn't ship `libnuma`, which vLLM's native CPU extension links against.
   The failed import cascaded into a confusing, unrelated-looking
   downstream `AttributeError` on `torch.ops._C.init_cpu_memory_env` —
   worth remembering that a missing native `.so` can surface as a Python
   attribute error several layers away from the real cause. Fixed with
   `apt-get install libnuma1`.
6. **`ValueError: Available memory ... less than desired CPU memory
   utilization`.** On vLLM's CPU backend, `--gpu-memory-utilization`
   (default ~0.9) is repurposed to mean "fraction of *host* RAM to
   reserve," not GPU VRAM — a real semantic overload of a flag name that
   looks GPU-specific. It overshot the kind node's actual free memory
   (~4.15GiB of 7.75GiB total). Fixed with `--gpu-memory-utilization 0.4`.
7. **`InvalidCxxCompiler: No working C++ compiler found (g++)`.** vLLM still
   runs `torch.compile`/inductor for certain fused ops (gelu, rms_norm)
   even with `--enforce-eager` — my first guess, that `--enforce-eager`
   alone would avoid any compile path, was wrong, and I said so explicitly
   when the error proved it. `--enforce-eager` only skips CUDA-graph
   capture, not this separate compile path. Fixed by installing `g++`
   directly (`apt-get install g++`).

Final working `k8s/Dockerfile` ENTRYPOINT:

```
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.4 --enforce-eager
```

### Task #1 — CLOSED

Environment-adaptation note (GPU requests/tolerations dropped, CPU-only real
vLLM instead of `vllm-metal`) documented inline as comments in both
`k8s/Dockerfile` and `k8s/deployment.yaml`, not just in this log.

## Entry: Task #2 — Service + Ingress

**Date:** 2026-08-14

Added `k8s/service.yaml` (ClusterIP, port 80 → pod's `http`/8000) and
`k8s/ingress.yaml` (`ingressClassName: nginx`, path `/` prefix, a generous
`proxy-read-timeout: "300"` to accommodate slow CPU inference rather than
letting nginx's default timeout mask a working-but-slow backend as a
failure).

Required installing `ingress-nginx`'s official kind-flavored manifest. That
manifest's controller pod has a `nodeSelector` on
`ingress-ready=true` on the control-plane node — our `kind-config.yaml`
hadn't set that label at cluster-creation time, so the controller pod sat
unscheduled. Fixed by labeling the existing node directly
(`kubectl label node vllm-local-control-plane ingress-ready=true`) rather
than tearing down and recreating the cluster just to add a label at
creation time.

Verified genuinely end-to-end, not just "resources created": `curl
localhost:8080/health` → 200, then a real `/v1/chat/completions` request
through the full path (host port mapping → ingress-nginx → Service → pod →
real vLLM inference) returned an actual model response, confirmed by
reading the JSON body rather than just the HTTP status code.

### Task #2 — CLOSED

## Entry: Task #3 — Helm chart

**Date:** 2026-08-13/14

Built `k8s/vllm-chart/` (`Chart.yaml`, `values.yaml`,
`templates/{deployment,service,ingress}.yaml`), parameterizing: `model`,
`maxModelLen`, `replicaCount`, `gpu.count` (0 by default, since this
environment has no GPU node — set `>0` genuinely adds real
`nvidia.com/gpu` requests/limits and a toleration for an actual GPU
cluster, not a placeholder that does nothing), `gpuMemoryUtilization`/
`enforceEager`, shm size, resource requests/limits, and ingress settings.

**Real, stated limitation, not glossed over:** the model is baked into the
image at build time (a consequence of task #1's offline-startup fix), so
changing `model` via `values.yaml` only actually works if `image.tag` also
points at an image that has that model's weights baked in. This is written
directly into `values.yaml`'s own comments rather than left implicit.

`helm lint` clean; `helm template` output diffed against the hand-written
manifests from tasks #1/#2 before running `helm install`, to confirm the
chart genuinely reproduces the same resources rather than trusting the
templating blindly. The old raw manifests (`k8s/deployment.yaml`,
`service.yaml`, `ingress.yaml`) were then deleted, superseded by `helm
install vllm-cpu-demo k8s/vllm-chart`.

Verified genuinely end-to-end post-install, not just "helm said deployed":
hit a real transient 503 right after install. nginx's own controller logs
showed "does not have any active Endpoint" — a timing race between the
freshly-recreated `Ingress` object and the new pod becoming ready, not a
config bug. Diagnosed via those logs rather than assumed, then confirmed
resolved on retry: `/health` → 200 and a real `/v1/chat/completions`
response through the Helm-managed Service/Ingress.

### Task #3 — CLOSED

## Entry: Task #4 — Prometheus ServiceMonitor → existing Grafana

**Date:** 2026-08-13/14, paused for a machine restart, resumed and closed
2026-08-14

### The real design decision

The existing Phase 1 observability stack
(`observability/docker-compose.yml`) is plain Prometheus with static
`scrape_configs` — no Prometheus Operator installed, so it has no concept of
a `ServiceMonitor` CRD at all. Two paths existed: quietly add the K8s pod as
a static scrape target in `prometheus.yml` (fast, but not what the master
plan actually asks for — task #4 explicitly names `ServiceMonitor`), or
install a genuine, lightweight Prometheus Operator inside `kind`, create a
real `ServiceMonitor`, and bridge its output into the *same* existing
Grafana as a second data source. Chose the real path deliberately — the
whole point of this task is demonstrating K8s-native service discovery, not
just getting a metric to appear.

### What was built

- Prometheus Operator installed pinned to `v0.93.1` (not `main`, for
  reproducibility) via `kubectl apply --server-side -f
  https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.93.1/bundle.yaml`.
- `k8s/monitoring/prometheus-rbac.yaml` (ServiceAccount + ClusterRole +
  ClusterRoleBinding) and `k8s/monitoring/prometheus.yaml` (a `Prometheus`
  CR, deliberately minimal — not the full `kube-prometheus-stack`, no
  bundled Alertmanager/node-exporter/kube-state-metrics/second Grafana.
  Sized small (cpu 100m/500m, memory 256Mi/512Mi request/limit) after
  actually checking node headroom first (~3.8GiB of memory requests
  unclaimed on the kind worker at the time), not guessed.
- `k8s/vllm-chart` updated: `templates/service.yaml` gained
  `metadata.labels` (previously only `spec.selector` existed — a
  `ServiceMonitor`'s own `matchLabels` selector needs labels on the
  `Service` object itself to find it, a distinction that isn't obvious
  from the Service spec alone); new `templates/servicemonitor.yaml`, gated
  behind `values.yaml`'s `serviceMonitor.enabled` (defaults `false` so a
  fresh `helm install` never fails against a cluster where the Operator's
  CRDs don't exist yet — must be explicitly enabled via `--set
  serviceMonitor.enabled=true` only *after* the Operator is installed;
  sequencing matters here and isn't enforced by Helm itself).
- Added a second Grafana datasource ("Prometheus (K8s)", fixed `uid`
  `prometheus-k8s-vllm`) to the existing provisioning file
  (`observability/grafana/provisioning/datasources/prometheus.yml`),
  following that file's own established pattern of fixed `uid`s — a Phase 1
  lesson (see that file's own comment) about unfixed `uid`s previously
  causing a silent "No data" bug in Grafana. Bridge: `kubectl port-forward
  --address 0.0.0.0 svc/prometheus-operated 9092:9090`, reachable from the
  Grafana Docker container via `host.docker.internal:9092`. This bridge is
  a foreground process, not a cluster resource — it does not survive a
  terminal closing, a machine restart, or (found live this session) simply
  not having been started at all, and needs to be re-run by hand each time
  it's needed.

### Two real "looks done but isn't" gaps caught by verifying rather than assuming

1. Grafana's own container logs showed the datasource loaded successfully
   (`msg="inserting datasource from configuration"
   name="Prometheus (K8s)"`) after the first provisioning change, but the
   user reported not seeing it in the UI — turned out to be a stale
   browser tab from before the Grafana restart, not a real config problem.
2. Later in the same session, after a full machine restart, the datasource
   "Test" button failed with `dial tcp ... connect: connection refused`.
   Checked directly (`ps aux | grep port-forward`, `lsof -iTCP:9092`)
   rather than assuming the earlier bridge was still up — it genuinely was
   not running at all (no process, nothing listening). This is expected
   behavior for a foreground port-forward, not a bug, but it's a real
   recurring operational gotcha for this setup worth flagging: the bridge
   needs to be manually restarted any time the terminal session, Docker
   Desktop, or the machine itself restarts.

### Final verification, genuinely end-to-end

With the port-forward confirmed listening (`lsof` showed `kubectl` bound to
`*:9092`):

- `curl localhost:9092/api/v1/query?query=up` → 200, real result.
- From inside the `vllm-grafana` container itself (`docker exec ... wget
  http://host.docker.internal:9092/...`) → same real result, confirming the
  actual path Grafana uses, not just host-side reachability.
- Queried `vllm:num_requests_running` (the same metric name Phase 1's
  existing dashboard already uses) through the bridge → real value
  returned from the K8s-sourced series, labeled with real K8s service
  discovery metadata (`namespace=default`, `pod=vllm-cpu-demo-...`,
  `service=vllm-cpu-demo`).
- In the Grafana UI, opened the existing "Tokens/sec vs Concurrency" panel
  (the one existing Phase 1 panel that already queries
  `vllm:num_requests_running`) and switched its datasource to "Prometheus
  (K8s)" — it rendered, showing a flat `0` line. That flat zero is correct,
  not a failure: the K8s pod is idle, and a broken connection would have
  shown "No data," not a real zero. Confirmed a real Phase 1 dashboard
  panel can render live K8s-sourced metrics through the new Operator
  pipeline, which is the actual ask behind task #4.

### Task #4 — CLOSED

## Entry: Task #5 revisited — object-store durability extension, and an 8th real bug

**Date:** 2026-08-15 through 2026-08-17

### Why revisit a closed task

Task #5's core ask (harness as a K8s Job, JSONL to PVC) closed cleanly on
2026-08-15/16 — see the entries above. But the master plan's own wording is
"JSONL to PVC *or object store*," and `bench-results-pvc` is backed by
`kind`'s default `local-path` StorageClass — real storage, genuinely
survives individual pod/Job deletions, but **not** `kind delete cluster`,
since under the hood it's just a directory inside the disposable node
container's own filesystem. Chose to close that gap for real rather than
leave it as a stated-but-unaddressed limitation, and to use the exercise as
a full from-scratch revision of task #5 (deliberately recreating the
cluster to force a genuine "start from nothing" redo, not just a mental
exercise).

### The real fix: `hostPath` + `kind`'s `extraMounts`, not a fancier PVC

A naive "just add MinIO" wouldn't have solved anything — MinIO's own PVC
would use the exact same `local-path` StorageClass, inheriting the identical
durability ceiling. The actual fix has nothing to do with Kubernetes storage
abstractions at all: `k8s/kind-config.yaml`'s worker node now has an
`extraMounts` entry bind-mounting `k8s/minio-data/` (a real directory on the
Mac's own disk) to `/mnt/minio-data` inside the node container. Verified
this is a genuine two-way bind (not two synced copies) by writing a file
from the Mac side and reading the identical content back via `docker exec`
into the node. `extraMounts` can only be set at cluster-creation time, which
meant tearing down and recreating `vllm-local` entirely.

### Full rebuild from a blank cluster

`kind delete cluster --name vllm-local` → `kind create cluster --config
k8s/kind-config.yaml` (with the new `extraMounts`), then rebuilt every piece
from tasks #1-#4: reloaded both images (`vllm-cpu-demo:v1`,
`vllm-bench-harness:v1` — these persist in Docker Desktop's own image store
independent of the `kind` node, so only needed re-loading, not rebuilding),
relabeled the control-plane node `ingress-ready=true`, reinstalled
`ingress-nginx`, reinstalled the Prometheus Operator (pinned `v0.93.1`,
same as before) + `k8s/monitoring/*.yaml`, and `helm install`ed the chart
with `--set serviceMonitor.enabled=true`.

### Bug #8: a real scheduler race, distinct from task #3's endpoint race

Post-rebuild, `curl localhost:8080/health` failed with `Recv failure:
Connection reset by peer` — not the task #3 "no active Endpoint" 503, a
different failure signature entirely. Diagnosed properly rather than
retried blind: confirmed from *inside* the cluster that both the
`ingress-nginx` Service and the `vllm-cpu-demo` pod answered `200`
correctly (via a throwaway debug pod), which narrowed the fault to
specifically the host-port-forwarding hop. Checked `docker port` on both
node containers directly — only `vllm-local-control-plane` has the
`8080->80` mapping (per `kind-config.yaml`'s `extraPortMappings`, which is
only declared on that node); `vllm-local-worker` has none at all. Then
checked *which node the ingress-nginx-controller pod actually landed on*:
`vllm-local-worker` — the wrong one.

Root cause: the upstream `kind`-flavored `ingress-nginx` manifest declares
`hostPort: 80/443` on the controller container (binds directly to whatever
node it schedules onto) and a toleration for the control-plane's taint (so
it's *allowed* to land there) — but no `nodeSelector` actually forcing it
onto that specific node. The first time this was built (2026-08-13/14), it
happened to schedule onto the control-plane node by scheduler luck. This
time, with two otherwise-empty nodes to choose from, it landed on the
worker instead — which has no host port mapping at all, so Mac's
`localhost:8080` was forwarding into a control-plane container port that
nothing was listening on anymore. That's exactly why the failure was a TCP
*reset* rather than a timeout or connection-refused.

Fixed by patching the Deployment directly to add the missing constraint:

```
kubectl patch deployment ingress-nginx-controller -n ingress-nginx \
  --type merge -p '{"spec":{"template":{"spec":{"nodeSelector":
  {"kubernetes.io/os":"linux","ingress-ready":"true"}}}}}'
```

**Lesson for any future cluster recreate:** this manifest's node placement
was never actually deterministic — it worked before by chance, not because
anything pinned it. Apply this `nodeSelector` patch as a standard step
immediately after installing `ingress-nginx`, not just when it happens to
break.

Verified fully end-to-end after the patch: `/health` → `200`, and a real
`/v1/chat/completions` response with genuine generated text through the
full `localhost:8080` → ingress → Service → pod path.

### Deploying MinIO and wiring the harness to it

`k8s/minio-deployment.yaml` (image `minio/minio:latest`, args `server /data
--console-address :9001`) + `k8s/minio-service.yaml`. The Deployment is
pinned with `nodeSelector: kubernetes.io/hostname: vllm-local-worker` —
deliberately, to avoid the exact same bug class as bug #8: the `hostPath`
volume only exists on the worker node kind-config's `extraMounts` targets,
so an unpinned pod could land on the control-plane node and silently get an
empty (or freshly-created, non-`Directory`) mount instead. Confirmed
`.minio.sys` appeared for real in `k8s/minio-data/` on the Mac disk, and the
S3 API answered `200`.

Created a `bench-results` bucket via a throwaway `minio/mc` pod. Updated
`vllm-benchmark/Dockerfile.harness` to add the `mc` client, and added a new
`vllm-benchmark/run_and_upload.sh` wrapper that runs `run_bench.py` then
`mc cp`s the PVC's `runs/*.csv`/`*.jsonl` into the bucket.
`k8s/benchmark-job.yaml` updated with `--variant k8s-job-minio` and
`MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` env vars.

**Real build gotcha:** rebuilding the harness image directly from
`~/Documents/vllm-benchmark` failed repeatedly (`transferring dockerfile:
2B` / `no such file`) across several invocation styles, even though the
original unedited image had built fine from the same location days
earlier. Root cause: `~/Documents` is iCloud Drive-synced, and Docker
Desktop's build engine was getting a stale/incomplete view specifically of
the *just-edited* Dockerfile. Fixed by copying the build context to `/tmp`
(never iCloud-synced) and building from there. Apply this workaround again
if `Dockerfile.harness` or `k8s/Dockerfile` is edited and hits the same
symptom.

Reran the Job with the new image: real `mc cp` upload log (4 files, 217.40
KiB total, ~2 MiB/s), independently reverified via a second throwaway
`mc ls local/bench-results/` pod — `k8s-job-minio.csv` (4.0KiB),
`k8s-job-minio.jsonl` (105KiB), plus the older `k8s-job.csv`/`.jsonl` pair
riding along (harmless — `run_and_upload.sh`'s glob uploads everything
currently on the PVC, not just the newest variant).

### The durability proof, 2026-08-17

The actual point of this extension: prove MinIO's storage survives cluster
deletion while the PVC's does not. Predicted first, then executed:

```
kind delete cluster --name vllm-local
kind create cluster --config k8s/kind-config.yaml
kubectl apply -f k8s/minio-deployment.yaml -f k8s/minio-service.yaml
kubectl get pvc bench-results-pvc
kubectl run mc-check --rm --restart=Never --image=minio/mc --command --attach -- \
  sh -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc ls local/bench-results/"
```

Result, exactly as predicted:
- `kubectl get pvc bench-results-pvc` → `Error from server (NotFound)`.
  `benchmark-pvc.yaml` was never reapplied, and even if it had been, a fresh
  PVC would start empty — the point stands either way.
- `mc ls local/bench-results/` → all 4 files reappeared with their
  **original** timestamps (`23:15:33`, not a fresh write) and unchanged
  sizes, with zero re-running of the benchmark and zero re-upload.

This is genuine, live proof that in this topology MinIO's durability comes
entirely from the `hostPath`/`extraMounts` bind to the Mac's real disk, not
from anything Kubernetes-native — and that the PVC does not survive cluster
deletion, confirming the gap this extension was built to close. The
architectural point for a real cloud environment: an actual S3/GCS bucket
would give the same durability property without needing a manual
bind-mount trick, since the object store's backing storage is inherently
off-node there.

After the proof, rebuilt the rest of the stack on the same fresh cluster
(images reloaded, `ingress-nginx` + the bug #8 nodeSelector patch, Prometheus
Operator + `k8s/monitoring/*.yaml`, `helm install ... --set
serviceMonitor.enabled=true`) and reverified fully end-to-end: `/health` →
`200`, a real `/v1/chat/completions` completion through ingress → Service →
pod, and `vllm:num_requests_running` returned from the in-cluster
Prometheus with genuine K8s service-discovery labels — confirming tasks
#1-#4 all still hold on the rebuilt cluster, not just task #5's extension.

### Task #5 (revised, with object-store durability) — CLOSED

## Entry: Task #9 — ArgoCD GitOps deploy

**Date:** 2026-08-17/18

### Structural decision: two Applications, not one

The master plan's own scope for this task is "deploy via ArgoCD," but a
real question came first: should MinIO (storage) and the vLLM chart
(serving) sync as one Application or two? Compared via a 10,000-ft diagram
before deciding — chose **two separate Applications**, deliberately, over
folding MinIO into the vLLM chart. Reasoning: shared infrastructure and a
serving workload shouldn't share one sync/rollback unit in a real platform
team, and this is the small-scale version of ArgoCD's real "app of apps"
pattern. `k8s/argocd/vllm-chart-app.yaml` (Helm source) and
`k8s/argocd/minio-app.yaml` (directory source, `include: "minio-*.yaml"`
so it doesn't try to apply `kind-config.yaml` or anything else in `k8s/`).

### Prerequisite: closing a real git/live drift before ArgoCD ever touched the cluster

`values.yaml` had `serviceMonitor.enabled: false` committed while the live
cluster had it manually `--set` to `true` at install time. Since GitOps
means "git is the only source of truth," this had to be fixed *first* —
otherwise ArgoCD's first sync would have silently deleted the live
ServiceMonitor to match the (stale) committed default. Fixed by flipping
the committed default to `true` before installing ArgoCD at all.

### Installing ArgoCD: two real, fixed bugs

`kubectl apply` (client-side) failed on the `ApplicationSet` CRD: `Too
long: may not be more than 262144 bytes` — client-side apply stores a full
last-applied-config annotation, and this CRD is large enough to blow the
262KB Kubernetes annotation limit. Fixed with `--server-side` (no
annotation stored). That left a field-manager conflict from the partial
first attempt (a few objects had already been created by the failed
client-side apply) — fixed with `--force-conflicts`, safe here since
nothing else was managing those fields.

### Migrating the existing manual Helm release

`helm uninstall vllm-cpu-demo` run deliberately before creating the
ArgoCD Application, to avoid dual ownership between Helm's own release
record and ArgoCD's independently-rendered manifests — standard practice
for adopting an existing release into GitOps, not a workaround.

### Bug: ArgoCD's default Helm release name isn't your old one

First sync created resources named `vllm-chart-*` (Deployment, Service,
Ingress, ServiceMonitor), not `vllm-cpu-demo-*`. ArgoCD defaults a Helm
Application's release name to the **Application's own name**, not
whatever release name was used with `helm install` manually. This wasn't
cosmetic — `k8s/benchmark-job.yaml` hardcodes `http://vllm-cpu-demo:80` as
its `--base-url`, which would have silently broken on the next benchmark
Job run. Fixed by pinning `spec.source.helm.releaseName: vllm-cpu-demo`
explicitly in `vllm-chart-app.yaml`. Re-synced; correct names came back,
old `vllm-chart`-named objects pruned cleanly by ArgoCD itself.

### Bug: host-RAM preflight failure — same class as task #1's bug #6, new trigger

Pod crash-looped: `Available memory on node 0 (3.06/7.75 GiB) ... is less
than desired CPU memory utilization (0.4, 3.1 GiB)`. Same underlying
mechanism as task #1's original bug #6 (`--gpu-memory-utilization` on
vLLM's CPU backend means "fraction of host RAM," not VRAM) — but a new
trigger this time: the worker node now permanently hosts Prometheus and
MinIO alongside vLLM, which it didn't during the original task #1 install.
Fixed: `gpuMemoryUtilization` lowered `0.4` → `0.3` in `values.yaml`,
committed and pushed. Watched `selfHeal` pick this up automatically with
zero manual `argocd app sync` — confirmed live in the ArgoCD UI
(port-forwarded `svc/argocd-server`), including correctly predicting
beforehand that a live `kubectl patch` instead of a git commit would just
get silently reverted.

### Bug: RollingUpdate needs 2x capacity, this node doesn't have it

The `0.3` fix triggered a rollout, visible live in the ArgoCD UI's resource
tree: two ReplicaSets, two pods. The *old* pod (pre-fix) actually
self-stabilized mid-investigation and came up genuinely healthy — a real,
unplanned data point (unlike task #1's original crash-loop, which never
self-resolved). The *new* pod stuck `Pending` for 6+ minutes:
`FailedScheduling: Insufficient memory`. Root cause: `gpuMemoryUtilization`
only controls vLLM's *internal* memory choice — it never touched the Pod's
declared Kubernetes `resources.requests.memory: 4Gi`, which is all the
scheduler actually looks at. Kubernetes' default `RollingUpdate` strategy
tries to schedule the new pod (wants `4Gi`) while the old one (holding
`4Gi`) is still running, and the only untainted node doesn't have room for
both at once. Fixed by adding `strategy: { type: Recreate }` to the
Deployment template — kills the old pod fully before creating the new one,
the standard real choice for a single-replica pod on a resource-constrained
node, not a hack.

### Bug: a misplaced field Kubernetes silently pruned instead of rejecting

The `Recreate` fix didn't actually take effect on the first attempt, even
though ArgoCD reported `"successfully synced (all tasks run)"` on every
sync — `autoHealAttemptsCount` reached **84** with the live Deployment
still showing default `RollingUpdate`/`25%`/`25%`. Root cause, found by
diffing `helm template` output line-by-line: `strategy:` had been added at
4-space indent, nesting it *inside* `selector:` (a `LabelSelector` object)
instead of as a sibling of `selector`/`template` under `spec:`. Kubernetes'
schema validation doesn't error on this — it silently prunes fields that
don't belong on a given sub-object, so the Deployment kept its default
strategy the entire time with no error anywhere in the chain. Fixed by
correcting the indentation; verified with a real dry-run apply against the
API server (not just YAML syntax) before pushing.

### Bug: KV cache starved by the same fix that solved the host-RAM bug

Once `Recreate` genuinely took effect, the old pod was killed and the new
one finally scheduled — and immediately crash-looped on a **different**
memory error: `0.38 GiB KV cache is needed ... available KV cache memory
(0.14 GiB)`. Direct callback to Phase 1 item #2's core mental model (fixed
weight/overhead cost subtracted from a shrinking pool): lowering
`gpuMemoryUtilization` to `0.3` shrank the whole memory pool, and since
model weights + overhead are a fixed cost regardless of the fraction, the
KV cache got the leftover — which was no longer enough to serve even one
request at `maxModelLen: 32768`. Fixed by lowering `maxModelLen` to `8192`
instead of pushing `gpuMemoryUtilization` back up on an already-tight
shared node — vLLM's own error message stated the real ceiling (`~12544`),
picked `8192` for real margin rather than hugging it.

### Final verification

`kubectl get application -n argocd` → both `vllm-chart` and `minio`
`Synced`/`Healthy`. Exactly one pod, `1/1 Running`, `0` restarts. Real
`/health` → `200` and a real `/v1/chat/completions` completion through the
full `localhost:8080` → ingress → Service → pod path — same standard used
for every task in this phase.

### Task #9 — CLOSED
