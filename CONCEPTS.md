# Concepts: scaling GPU replicas on queue depth without paying for cold starts twice

`gpuscaler` is a controller that watches how many requests are waiting across a pool of inference replicas and adds or removes replicas to hold that number near a target, with a cooldown so it does not flap and a warm standby so a scale-up does not take minutes. It runs against tokenbench mock replicas on a laptop, replays a bursty traffic trace, and reports the latency users saw against the GPU-hours paid, with and without the standby. This primer covers why queue depth is the right signal, what the KEDA model is, and why cold start is the whole problem.

## 1. Why not CPU or GPU utilisation

Classic autoscalers scale on CPU utilisation. For LLM serving, GPU utilisation is a poor signal: a replica serving one user at 68 tokens/s shows the GPU "busy" (the kernels run back to back) while doing 2% of the useful work it could; a replica at its concurrency limit shows the same number. Utilisation does not distinguish an idle-ish server from a saturated one, and it says nothing about users waiting.

**Queue depth** (requests admitted but not yet running, plus requests the gateway is holding) is the direct measure of unmet demand. It is zero exactly when every request is being served, it grows the moment the pool falls behind, and its rate of growth is the arrival rate minus the service rate: the quantity a scaler must correct. vLLM exports it per replica as `vllm:num_requests_waiting`; a gateway in front can add its own pending count. Little's law converts it into the latency users will see: queue time ≈ queue depth ÷ throughput.

The secondary signal is **per-replica concurrency** (`num_requests_running`) against the replica's known good operating point (servellm's curves: ITL starts climbing beyond ~16–32 concurrent on a T4 with the 1.5B model). Scaling on that keeps latency bounded before queueing even begins; this project uses queue depth as the primary trigger and concurrency as the scale-down guard.

## 2. The KEDA model

Kubernetes' HPA scales on metrics; KEDA (Kubernetes Event-Driven Autoscaling) adds "scalers" that read external metrics (a Prometheus query, a queue length) and drives the HPA with them. Its core arithmetic is:

```
desired_replicas = ceil( current_metric_total / target_per_replica )
```

For queue depth with a target of, say, 4 waiting requests per replica: 40 waiting across the pool → 10 replicas. KEDA clamps to `minReplicaCount`/`maxReplicaCount`, applies the HPA's stabilisation window (scale-down waits until the metric has been low for N seconds) and a cooldown after scaling, and can scale to zero when the metric is zero for `cooldownPeriod`. `gpuscaler` implements exactly this loop in-process, and the repo includes the equivalent `ScaledObject` manifest so the mapping to a real cluster is explicit. The controller is the thing being studied; Kubernetes is where it would live.

## 3. Cold start is the whole problem

A new CPU pod serves traffic in seconds. A new vLLM replica must: get a GPU scheduled (seconds to minutes in a cloud), pull a multi-gigabyte image, load weights (30 s for 1.5B from a warm disk, minutes for 70B from object storage), profile memory, capture CUDA graphs and compile (servellm measured ~90 s from launch to `/health` for a 1.5B model on a T4; a 7B model on a cold cloud VM is routinely 5–10 minutes). During that time the queue the scaler reacted to keeps growing. By the time the replica is ready the burst may be over, and the scaler then pays again to scale down.

Three mitigations, with costs:

- **Warm standby**: keep N extra replicas loaded and idle (or "sleeping": vLLM 0.31 has a sleep mode that offloads weights to CPU RAM and wakes in seconds). Costs idle GPU-hours continuously; buys near-instant scale-up up to N.
- **Faster cold start**: pre-pulled images, weights on local NVMe or in page cache, skipping CUDA-graph capture (`enforce_eager`, at a decode-speed cost). Costs engineering; shortens the gap but does not close it.
- **Predictive scaling**: scale on a forecast (time of day, trend) rather than the current queue. Costs forecast errors in both directions.

This project measures the first against the baseline and models the cold start explicitly (a replica that is "starting" accepts no traffic for a configurable delay, 90 s by default from the servellm measurement), because the mock replicas start instantly and would otherwise make scaling look free.

## 4. Flapping and the two time constants

A scaler that reacts to every blip adds replicas it will remove a minute later, paying cold start both ways. Two parameters tame it: a **scale-up cooldown** (do not add more replicas until the last addition has had time to become ready and absorb load; shorter than the cold start is pointless) and a **scale-down stabilisation window** (remove a replica only when the metric has been below target for the whole window, typically several minutes, because removing is cheap to delay and expensive to regret). The asymmetry is deliberate: scale up fast, scale down slow.

## 5. The cost side

GPU-hours = Σ over replicas of (time from launch to termination), including the cold-start time during which the replica earns nothing. A trace replay yields two numbers per policy: GPU-hours consumed, and the latency distribution users experienced (TTFT including queue wait). Plotting one against the other across policies (no scaling, reactive, reactive + warm standby) is the trade-off the FinOps conversation needs: "the standby costs X extra GPU-hours per day and removes the Y-second TTFT spikes at the top of each burst." Against a fixed-size pool sized for the peak, autoscaling's saving is the area between the peak-sized line and the replica curve.

## 6. What the experiment will show

Against a bursty trace (a quiet baseline, a 10x burst for a few minutes, a decay, repeated), with mock replicas that emulate a 90 s cold start and a per-replica capacity:

1. **Fixed pool sized for the peak**: flat latency, maximum GPU-hours.
2. **Fixed pool sized for the average**: cheap, with a queue that explodes at each burst.
3. **Reactive scaling on queue depth**: replicas follow the trace with a lag equal to the cold start; the latency spike at the start of each burst has that length.
4. **Reactive + one warm standby**: the spike shrinks to the time to absorb the first wave; GPU-hours rise by one replica's worth.

Timelines of replicas, queue depth, p95 TTFT per minute and cumulative GPU-hours, and one table of GPU-hours vs latency per policy. The mock replicas make this reproducible in minutes and the T4 numbers from servellm make the parameters (capacity, cold start) honest.

## Further reading

- KEDA documentation: *Scaling Deployments*, the Prometheus scaler, `ScaledObject` fields (`pollingInterval`, `cooldownPeriod`, `minReplicaCount`, advanced HPA behaviour).
- Kubernetes HPA documentation on stabilisation windows and scaling policies.
- vLLM docs: *Sleep Mode*, for the warm-standby mechanism inside one process.
- Little's law (tokenbench primer, section 4) for turning queue depth into latency.
