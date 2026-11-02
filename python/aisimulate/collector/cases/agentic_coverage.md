# Agentic workload coverage

This document defines collector coverage targets for agentic replay, the workload evidence behind them, and the checks required before collected data can be trusted. Shared ranges belong in [base operation cases](base_ops/); correlated model geometry belongs in [model cases](models/). Coverage is enabled by default, without an agentic toggle.

**Status:** implemented defaults and remaining targets are distinguished below. Implementation does not establish GPU feasibility, timing accuracy, or complete replay support. The workload snapshot is **2 November 2026**. Update the budget, implementation status, and qualification together when the source or workload changes.

The workload analysis covers **808 active NVIDIA configurations**, with AMD excluded. The source review covers **78 CUDA collector registry routes**, grouped into **36 operation families**. Public request records establish history demand and configuration envelopes; they do not expose every executed per-rank GPU forward. A missing exact anchor is not automatically a functional replay failure: interpolation, extrapolation, and declared fallback paths are separate contracts.

Token shorthand uses binary multiples: **1K = 1,024 tokens; 1M = 1,048,576 tokens**. Byte sizes use MiB/GiB. Token counts, tensor elements, and bytes must not be compared without conversion.

## Coverage targets

“Implemented” identifies enabled collector defaults. “Conditional target” identifies follow-up work requiring the named model, serving recipe, or native path. “Retain” means no broader range is established by the available evidence. None of these labels means that every declared shape has passed GPU qualification.

| Operation / setting | Budget | Status and rationale |
|---|---|---|
| Shared dense MHA/GQA context | **65,536, 131,072, 524,288 and 1,048,576** full-history anchors; **B × S ≤ 1,048,576** on vLLM, SGLang and TRT-LLM | **Implemented.** Preserve old anchors, geometry, precision and independent index/memory guards. The 64K/128K intervals serve supported Qwen3.5; the shared 1M tail is policy for eligible geometries, not evidence that every active model uses dense attention. |
| Ordinary fused MoE | Token anchors through **32,768 and 65,536** | **Implemented.** Local work can reach 32K; generic MoE can gather 16K local tokens across attention DP 4 into a 64K table query. These are configuration envelopes, not observed filled forwards. Preserve backend-specific token keys and TP/EP geometry. |
| SGLang MSA context | Per-request new-query anchor **32,768**; retain **131,072** aggregate new-token budget | **Implemented.** MiniMax settings permit a 32K local chunk. Native chunk capacity follows the planned query batch; query/history/batch and model limits remain joint constraints. |
| Standalone TRT-LLM MLA | Synthetic full history through **1,048,576**; retain native head/precision restrictions | **Implemented.** Size Scenario/RoPE capacity for the native cache manager instead of rejecting lengths above 163,840. Preserve YaRN factor 40 and original-position 4,096; charge the duplicate FP32 RoPE allocation to memory admission. This does not change any checkpoint's positional definition. |
| Ordinary TRT-LLM MoE completeness | Every requested shape must produce a measurement or a classified failure | **Implemented.** Dry-run the actual largest requested count; propagate tuning and execution failures. The existing task groups a whole token sweep, so a failure in its initial dry run fails that group before smaller counts run. No task/schema change is introduced. |
| KDA verification | Query widths **{2,3,4,5,6,8}** tokens/request; preserve verifier batch/state geometry | **Implemented.** Active vLLM Kimi settings require exact width anchors 3,5,6. SGLang's checked DSpark draft-token knob already represents width; do not add one twice. |
| NCCL all-gather / reduce-scatter | Default benchmark payloads through **1 GiB** | **Implemented.** Local 16K × DP 4 × hidden 6,144 × BF16 2 bytes produces a 768 MiB gathered-buffer key. The shared NCCL sweep also extends all-reduce/all-to-all; dtype and topology remain separate axes. |
| Ordinary MoE above 64K | **131,072** global token anchor for qualified MiniMax recipes | **Conditional target.** A vLLM decode configuration permits 8,192 local scheduled tokens × DP 16. Native AgRs is logged, but executed batch/verifier tuples and the final selected compute table still need verification. |
| NCCL large DP groups | Payloads through **2 GiB**, including a **16-rank** topology | **Conditional target.** The same MiniMax allowance gives 1.5 GiB gathered BF16 activations. More bytes on an eight-GPU table do not establish 16-rank coverage. |
| SGLang MLA/DSA/MSA generation history | Add **786,432, 917,504 and 1,048,575** past-history anchors within existing aggregate budgets | **Implemented.** GLM/MSA histories exceed 720,896. Model positional limits still apply, and a legal boundary depends on native verifier width. MLA applicability depends on the routed model; active Kimi configurations use vLLM. |
| vLLM DSA context | **B × q ≤ 32,768**; retain separate 1M aggregate full-history bound | **Implemented for native-capable recipes.** The query list already reached 32K; the old 16K aggregate cap removed these points. This does not establish an observed active GLM 32K forward. |
| SGLang DSA / GLM sparse queries | Query anchor **32,768**; retain **131,072** aggregate new-token cap | **Implemented for matching native capacity.** Size the default prefill harness from planned queries; preserve explicit overrides and independent model/index/pool limits. Runtime violations fail visibly. |
| WideEP expert compute | TRT global-token axis through **65,536**; local-axis expansion remains conditional | **Inherited on TRT** from the shared MoE token list; its existing global-key mapping is unchanged. vLLM/SGLang WideEP use separate grids. A global 64K query is not 64K on each EP rank; routing, receive capacity, and topology still require qualification. |
| DSV4.1 module manifests | Legal joint cases through **1M total history** and **32K new query**, with exact stage identities | **Conditional target.** Request history reaches 992,350, while the default manifest is B ∈ {1,2}, q ≤ 1,024, prefix ≤ 1,024 and the route is unverified. Speculation/DP/CP restrictions require separate work. |
| Dense / MLA / DSA / MSA decode aggregates | Retain family-specific budgets until a **correlated per-rank tuple census** establishes a replacement | **Retain.** Client concurrency and global interval overlap are not native GPU batches. Static verifier expansion is not physical KV replication. |
| GEMM, scale operations, recurrent context, MLA BMM, mHC, encoder, and most A2A paths | Retain the operation-specific ranges listed below | **Retain.** These use new-token, batch, geometry, or message axes. Long retained history alone does not increase them. |

## Workload scope and measurement method

The [InferenceX catalog](https://inferencex.semianalysis.com/) snapshot at **2026-11-02 19:52:39 UTC** contained **1,074 active/latest points = 808 NVIDIA + 246 AMD + 20 TPU**. AMD is excluded. TPU is a separate non-AMD stratum: 11 Qwen points exposed request timelines and server metrics; nine Kimi points returned 404 for both and the chart fallback. The CUDA targets use NVIDIA data only.

There are seven active display families and eight database model keys. The GLM-5.2 display family includes 73 `glm5.2` and 17 `glm5.3` NVIDIA points. Deprecated DeepSeek-V4-Pro is excluded; GLM5.3-Flash has no active points. Catalog churn changes this denominator independently of AISim feature support.

All 808 NVIDIA points exposed request timelines and server metrics. There were **12,875,136 request rows**, of which **12,764,189** had valid input and actual-output lengths. Profiling contributed 11,916,367 rows / 11,914,755 valid pairs; warmup contributed 958,769 / 849,434. Missing lengths are excluded, not imputed. No row was marked cancelled; that does not independently certify successful execution.

The following statistics pool request observations across configurations. Replayed source requests can appear repeatedly: these are not deduplicated production-traffic percentiles. Quantiles use linear interpolation at sorted index `(n − 1) × percentile`. Maximum history is computed **within each request** as input plus actual output, across warmup and profiling; marginal maxima are never added.

| Display family | NVIDIA points | Profiling ISL p95 | Profiling ISL p99 | Maximum ISL, all phases | Profiling actual OSL p95 | Maximum same-request history | History coverage target |
|---|---:|---:|---:|---:|---:|---:|---:|
| Muse-Spark-Open | 71 | 467,180 | 724,896 | 851,600 | 4,363.7 | [851,702](https://inferencex.semianalysis.com/inference/agentic/446222) | 1,048,576, subject to exact geometry |
| DeepSeek-V4.1-Flash | 140 | 441,092 | 710,822 | 987,557 | 3,960 | [992,350](https://inferencex.semianalysis.com/inference/agentic/445573) | 1,048,576 |
| Kimi-K3 | 82 | 445,843 | 720,529 | 950,778 | 3,837 | [950,779](https://inferencex.semianalysis.com/inference/agentic/446541) | 1,048,576 |
| MiniMax-M3.1-Flash | 186 | 469,306 | 734,613 | 952,822 | 4,019 | [959,333](https://inferencex.semianalysis.com/inference/agentic/447062) | 1,048,576 |
| GLM-5.2 display group | 90 | 448,395.5 | 718,255 | 950,587 | 3,946 | [959,120](https://inferencex.semianalysis.com/inference/agentic/446251) | 1,048,576 |
| Qwen-3.5-397B-A17B | 201 | 224,398 | 248,707.17 | 256,442 | 3,610 | [256,582](https://inferencex.semianalysis.com/inference/agentic/445216) | 262,144 |
| Qwen3.8-Flash-Next | 38 | 222,926.45 | 248,353.89 | 256,823 | 3,695 | [256,939](https://inferencex.semianalysis.com/inference/agentic/446962) | 262,144, after model support |

Maximum observed actual OSL is 59,903 in every display family; it did not necessarily occur with the longest prompt. Kimi's largest history is a warmup request at 950,779, exceeding its profiling maximum of 938,867. Profiling-only coverage would miss it.

**Data-quality exception:** [point 438947](https://inferencex.semianalysis.com/inference/agentic/438947) reports 522 completed profiling requests and 483,232 output tokens, but its timeline and chart expose 517 requests and 467,655 output tokens. The cause is unknown. The other 818 available non-AMD timelines matched published actual-output means within 0.0001. These are exposed-data extrema, not guaranteed true maxima for every run.

The pinned [Weka source corpus](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/23f152f6f0f9399a85901b89a6458def0ef16729/traces.jsonl) provides an independent tail check: 98,827 requests across 393 traces, including 42,029 subagent requests; input p95 682,880; maximum input 989,824; maximum same-request input plus target output 996,579. Its input length is a hash-block proxy (`blocks × 64 tokens/block`) and output is requested, not actual. It supports a 1M tail policy but does not replace run-specific measurements.

To repeat the measurement method:

1. Freeze the frontend model catalog and query `/api/v1/benchmarks?model=<display-name>`. Select current default-category, non-feature-gated families and their latest logical curves; retain all `agentic_traces` points without Pareto, framework, or offload filtering. Partition hardware vendors explicitly.
2. Fetch `/api/v1/request-timeline?id=<point-id>` and `/api/v1/trace-server-metrics?id=<point-id>`; use `/api/v1/request-chart-data?id=<point-id>` only as an identified diagnostic fallback. Record source URL, fetch time, response status, and SHA256 outside the repository.
3. Partition warmup/profiling, exclude invalid lengths per field, and require both lengths for joint history. Check point-ID uniqueness, phase totals, pooled counts, and published actual-output means. Preserve discrepancies rather than inventing rows.
4. Inspect native resolved server settings and model identity separately from CLI requests. Effective settings were available for 801/808 NVIDIA points: six Kimi points lacked logs, and MuseSpark TRT point 446733 exposed only initial CLI/model-loading output. At role granularity, 15 selected logs remained unknown, including eight truncated TRT decode logs whose points had usable prefill settings. Point-level coverage does not establish every serving role's settings. Unknown values remain unknown.
5. Keep source-corpus statistics, run observations, configuration envelopes, and actual per-rank forward traces separate. Future catalog changes require a new snapshot and denominator review.

## Budget equations and units

| Symbol | Meaning |
|---|---|
| `I_i`, `O_i`, `H_i` | Observed input, actual output, and endpoint history of request i: `H_i = I_i + O_i` |
| `B_p`, `B_d` | Requests in one prefill/decode forward on one attention rank |
| `B` | Homogeneous collector request batch: `B_p` or `B_d`, according to phase |
| `q_i`, `p_i`, `S_i` | New query tokens, retained prefix, and visible history: `S_i = p_i + q_i` |
| `q`, `p`, `S` | Homogeneous per-request values in a rectangular collector case |
| `X` | Local new-token work: `Σ q_i`, or `B_p × q` for homogeneous prefill |
| `D` | Attention-DP ranks gathered by the particular consumer |
| `V` | Target verification width; the legacy/static convention uses `nextn + 1` |
| `L` | Past-KV tokens; the one-token decode convention uses total `S = L + 1` |
| `q_native`, `P_max` | Query tokens executed by the native verifier, and the native model's maximum supported positions |
| `U`, `A` | Coverage budget in the filter's units; chosen history-anchor spacing |
| `h_q`, `h_kv`, `d` | Local query heads, local KV heads, and head dimension |
| `h`, `b` | Activation hidden width and bytes per element; `b` also denotes cache-element bytes when explicitly stated |
| `G`, `M_est` | Actual device memory and estimated native peak allocation, in bytes |
| `a`, `c`, `M` | GEMM token divisor, sequence-split divisor, and resulting GEMM row count |

### History ceiling

Choose a model-family tail anchor from joint observed lengths:

`H_anchor(m) = A × ceil(max_i(H_i) / A)`.

Here `max_i(H_i)` includes warmup and profiling for family m. `A = 65,536` is a grid-design choice, not a measured hardware alignment or percentile.

- Largest observed NVIDIA history: `65,536 × ceil(992,350 / 65,536) = 65,536 × 16 = 1,048,576`.
- Qwen3.5: `65,536 × ceil(256,582 / 65,536) = 65,536 × 4 = 262,144`.
- **16 and 4 count anchor intervals**, not batches, GPUs, or speculative tokens.

Individual rounding gives a smaller MuseSpark anchor; a shared 1M tail is an explicit simplification subject to its missing geometry. Clamp or replace any proposed boundary according to the actual model/runtime position limit. A context case may reach `p + q = max_position_embeddings`; a decode past-length case normally requires `L < max_position_embeddings`.

### Dense context: full history and aggregate positions

For a rectangular dense collector case:

`U_context ≥ B_p × S`.

The [dense consumer](../../../../crates/core/src/perfmodel/operators/attention.rs) queries full `S = p + q`, then applies the cached-prefix discount:

`estimated context latency = table(B_p, S, geometry) × (S² − p²) / S²`.

For example, `p = 32,768` cached tokens and `q = 128` new tokens require the `S = 32,896` lookup; per-request GEMM/recurrent work still uses only 128 new tokens. Small new queries therefore still need long-history anchors.

The previous budget `655,360 = 16 × 40,960` represents **16 synthetic lanes × 40,960 history tokens/lane**; it is not an all-workload percentile. The enabled budget `1,048,576 = 1 × 1,048,576` is **one request × the 1M history anchor**. It is the minimum aggregate cap admitting that B1 point, not proof of adequate multi-request coverage.

| Full history S | Maximum B from a 1,048,576 token-position cap |
|---:|---:|
| 65,536 | 16 |
| 131,072 | 8 |
| 262,144 | 4 |
| 524,288 | 2 |
| 786,432, optional | 1 |
| 1,048,576 | 1 |

Geometry, index, batch-list, native-backend, and memory constraints can admit fewer cases. Raising the cap also makes existing anchors eligible at larger batches, such as B4 × 262,144.

Supported Qwen3.5 dense layers need 64K/128K interval anchors; their model capacity is 262,144, so B1 coverage does not require a 1M aggregate cap. GLM, MSA, and DSV4.1 history does not establish generic dense demand. Kimi uses [native MLA modules with a vLLM granular fallback](../../src/aisimulate_core/sdk/models/kimi_k3.py); SGLang uses the separate MLA table. Kimi's draft dense attention is a generation path.

Kimi's [declared fallback TP sizes](models/KimiK3ForConditionalGeneration_cases.yaml) 1/2/4/8 produce MHA heads 96/48/24/12 at head dimension 128. Every one fails the existing KV-element guard at 1M. TP 8, B1, S = 524,288 survives; S = 786,432 and S = 1,048,576 do not. A new shared 1M anchor is a policy for eligible geometries, not completion of Kimi fallback coverage.

### New-token work and MoE: apply DP once

For homogeneous prefill:

`X = B_p × q`; `X_MoE = X × D`.

[GLM point 444870](https://inferencex.semianalysis.com/inference/agentic/444870) resolves a global launch chunk of 65,536 into local 16,384 tokens with attention DP 4:

`X_MoE,envelope = 16,384 × 4 = 65,536 tokens`.

**16,384** is the effective local allowance; **4** is the gather-group size; **65,536** is the derived table-key envelope, not a measured filled forward. The [native SGLang DP hook](https://github.com/sgl-project/sglang/blob/07c8f7294daa12087e969c89a11f773d36b9d42d/python/sglang/srt/arg_groups/parallel_hook.py#L191) divides the global chunk by DP. Multiplying the original global 65,536 by DP 4 again double-counts the split. Local 32,768 × DP 2 gives the same envelope. The [generic MoE consumer](../../../../crates/core/src/perfmodel/operators/moe.rs) applies this multiplication exactly once; specialized MegaMoE consumes local tokens instead.

SGLang can admit its first chunk above `max_prefill_tokens`, so `min(chunked_prefill_size, max_prefill_tokens)` is not a general local limit. The checked native [input-cap guard](https://github.com/sgl-project/sglang/blob/07c8f7294daa12087e969c89a11f773d36b9d42d/python/sglang/srt/managers/schedule_policy.py#L1240) applies without chunking; [chunk truncation](https://github.com/sgl-project/sglang/blob/07c8f7294daa12087e969c89a11f773d36b9d42d/python/sglang/srt/managers/schedule_policy.py#L1354) uses remaining chunk and memory capacity. Use resolved native settings and account separately for mixed steps, dynamic chunking, padding, and routing.

Ordinary GEMM follows:

`M = ceil(floor(X / a) / c)`.

`a` is `scale_num_tokens`; `c` is the sequence-split divisor. Neither automatically equals DP. If `X ≤ 32,768` and `a,c ≥ 1`, at most 32,768 GEMM rows are needed. [Compute-scale and scale-matrix](../../../../crates/core/src/perfmodel/operators/gemm.rs) use the same M. A 64K MoE target therefore does not imply a 64K GEMM target. [MLA BMM dispatch](../../../../crates/core/src/perfmodel/operators/op.rs) uses batch size, not token-major X.

| Family | Effective settings available | Largest declared local chunk/scheduler allowance | Largest potential generic-MoE group allowance |
|---|---:|---:|---:|
| Muse-Spark-Open | 70/71 | 32,768 | 32,768 |
| DeepSeek-V4.1-Flash | 140/140 | 32,768 | 65,536 |
| Kimi-K3 | 76/82 | 16,384 | 16,384 |
| MiniMax-M3.1-Flash | 186/186 | 32,768 | 131,072 |
| GLM display group | 90/90 | 16,384 | 65,536 |
| Qwen3.5 | 201/201 | 32,768 | 32,768 |
| Qwen3.8-Flash-Next | 38/38 | 16,384 | 16,384 |

These columns contain separate maxima across roles, not one combined configuration. The gathered column applies only where the selected consumer/path gathers those ranks; not every architecture uses generic MoE.

[MiniMax point 447020](https://inferencex.semianalysis.com/inference/agentic/447020) permits decode-side `8,192 × 16 = 131,072`: **8,192** is native `max_num_batched_tokens`, **16** is native DP, and `AgRsAll2AllManager` is logged. This motivates a conditional 128K global-key anchor. It does not prove a filled decode forward or selection of an ordinary fused-MoE table. Its prefill allowance is `16,384 × 4 = 65,536`.

### Sparse context: query and history are separate axes

DSA/MSA tables need `(B_p, q, p, geometry)`. Their new-token budget is `U_query ≥ B_p × q`; retained history is bounded separately.

A proposed one-request boundary case is:

`B_p = 1; q = 32,768; p = 1,048,576 − 32,768 = 1,015,808; S = 1,048,576`.

The factors are **one request**, **32,768 new tokens**, **1,015,808 seeded prefix tokens**, and **1,048,576 visible tokens**. This is a test boundary, not an observed forward. Prefix 1,048,575 cannot be paired with a 32K query under a 1M position limit.

For a full-index DSA layer, a useful work term is:

`F_index ≈ 2 × B_p × q × n_index × d_index × S`.

`F_index` is floating-point operations; **2** counts multiply and add; `n_index` and `d_index` are index-head count and dimension. GLM uses 32 and 128. Its selected-token limit of 2,048 does not cap the indexer's history scan. MSA scores approximately `ceil(S / 128)` blocks before selecting **16 blocks × 128 tokens/block = 2,048 tokens**. Index work and persistent KV still grow with S. See [DSA lookup](../../../../crates/core/src/perfmodel/perf_database/dsa.rs) and [MSA operator](../../../../crates/core/src/perfmodel/operators/msa.rs).

SGLang's aggregate `131,072 = 4 × 32,768` could admit **four requests × 32K new tokens/request** before other filters. This shows that adding a missing 32K query axis need not raise that scalar cap; it does not establish feasible B4 long-prefix execution.

### Decode: use correlated rank tuples

Let F contain actual per-rank forward records mapped through the exact entry point and collector key. For a rectangular past-history table:

`U_decode,target = max_{f∈F} [B_lookup(f) × L_anchor(f)]`.

`f` is one mapped forward; `B_lookup(f)` is its batch after the consumer's token/verifier conversion; `L_anchor(f)` is the selected anchor covering that forward's past history. Use total `S_anchor` instead if that collector stores total length. Do not combine the largest batch of one forward with another forward's longest history.

The [legacy/static runtime](../../../../crates/core/src/perfmodel/engine/runtime.rs) uses `B_lookup = B_d × V`, `V = nextn + 1`. Already-packed rank telemetry must not be multiplied again. KDA keeps `(B_d,V)` separately. Physical KV follows real histories; verifier width does not generally replicate KV storage V times.

Native verification must also satisfy `L + q_native ≤ P_max`. Here `q_native` is the actual query width executed by the verifier and `P_max` is native model capacity, not a collector budget. A past-history anchor of 1,048,575 leaves room for one query token in a 1M model; it is not automatically legal for wider verification.

The declared SGLang MLA/DSA budget can be decomposed as:

`37,486,592 = (13 × 4) × 720,896`.

- **13** is a sampled running-request count used to construct a coverage point, not a certified per-rank census.
- **4** is the static expansion for `nextn = 3`: three speculative additions and one base token.
- **52 = 13 × 4** is a token-basis lookup batch, not 52 independent physical request histories.
- **720,896** is its rounded history anchor.

This useful point is not a workload-wide statistical budget. Dense 16,777,216 and generic module 33,554,432 budgets likewise require joint-tuple validation. Dividing them by a 1,048,576 total-length anchor gives 16 and 32 rectangular requests; the arithmetic is not evidence of adequate live-batch coverage, and past-length tables have a one-token offset.

The snapshot contains up to 4,041 overlapping observed decode intervals across a DSV4.1 service and a client-concurrency setting of 10,888. These can span multiple workers/ranks and include timing uncertainty. An envelope using final history throughout each interval and excluding missing lengths is not a physical KV or GPU-forward measurement. It cannot justify a multi-billion-token collector budget.

### KDA verification width

For the verified vLLM Kimi path:

`V = N_spec + 1`.

`N_spec` is configured speculative tokens; **1** is the base target token. The pinned [native vLLM KDA metadata](https://github.com/vllm-project/vllm/blob/e4a5b5537344b103ed10de7764378ef09604ef3d/vllm/models/kimi_k3/nvidia/kda_metadata.py#L352) passes `spec_query_len = 1 + num_speculative_tokens`. Settings `{2,3,4,5,7}` yield widths `{3,4,5,6,8}`. The [collector widths](base_ops/kda.yaml) now include `{2,3,4,5,6,8}`, adding **3,5,6** to the previous grid. They are target query widths, not accepted-token counts.

Explicit width settings were available for 62/82 Kimi points; 50/62 require the newly added exact widths 3/5/6. The other 20 have insufficient evidence, not proven disabled speculation. In the checked SGLang DSpark path, [`speculative_num_draft_tokens` already represents V](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/speculative/dspark_components/dspark_config.py#L51). Preserve verifier batch, adaptive/ragged execution, state layout, and native kernel identity separately from width coverage. The [KDA consumer](../../../../crates/core/src/perfmodel/operators/mamba.rs) recovers request batch and queries `(B_d,V)`.

### Communication bytes and topology

For BF16 activation gathering on a non-DeepEP path:

`bytes_local = X × h × b`; `bytes_table = X × D × h × b`.

`X` is local scheduled tokens, `D` is gather-group size, `h` is hidden elements/token, and `b` is activation bytes/element. The [NCCL consumer](../../../../crates/core/src/perfmodel/operators/communication.rs) and nccl-tests [all-gather](https://github.com/NVIDIA/nccl-tests/blob/afd59abab774a6a4b492eb344bd0c845a0b7bb4d/src/all_gather.cu#L9)/[reduce-scatter](https://github.com/NVIDIA/nccl-tests/blob/afd59abab774a6a4b492eb344bd0c845a0b7bb4d/src/reduce_scatter.cu#L9) use the gathered-buffer key. This differs from local send size and physical link traffic.

- **GLM:** `16,384 × 4 × 6,144 × 2 = 805,306,368 bytes = 768 MiB`. Factors: local tokens, DP ranks, hidden elements/token, BF16 bytes/element. Local send is 192 MiB, but the table key is **768 MiB**, above 512 MiB; the next power-of-two anchor is **1 GiB**. Point 444870 logs `moe_a2a_backend='none'`.
- **MiniMax:** `8,192 × 16 × 6,144 × 2 = 1,610,612,736 bytes = 1,536 MiB = 1.5 GiB`. The target is **2 GiB with 16-rank topology**. The logged native AgRs manager supports investigating this envelope; exact source for build `8744dd2fb` is not publicly resolvable, so executed payloads remain unverified.
- **Units:** `1 MiB = 1,048,576 bytes`; `1 GiB = 1,073,741,824 bytes`. The [NCCL collector](../network/collect_nccl.py) accepts bytes but persists elements after dividing by dtype size.

NCCL's sweep upper endpoint is exclusive: including 1 GiB requires **1,073,741,825 bytes = 1 GiB + 1 byte**; including 2 GiB requires **2,147,483,649 bytes = 2 GiB + 1 byte**. These are iterator endpoints, not measured payloads.

The [HT A2A consumer](../../../../crates/core/src/perfmodel/operators/moe_a2a.rs) instead queries local tokens divided by attention TP; it does not apply the generic expert-compute DP multiplier. Global 64K/128K MoE envelopes therefore do not establish 64K/128K-per-rank HT demand. Native receive pressure still depends on EP, routing, top-k, hidden width, and dtype. All-gather can also carry routing weights/IDs as separate tensors, so the activation calculation is not a complete communication-buffer inventory.

## Model identity and functional limits

| Active family | Relevant paths | Qualification not solved by a larger budget |
|---|---|---|
| Qwen3.5 | 15 GQA and 45 GDN layers, MoE, projections, communication | GQA history target is 256K. GDN uses new q and recurrent state, not a scan of all history. Hybrid state/cache restoration is a separate functional contract. |
| GLM5.2 / GLM5.3 | DSA full/skip indexer, sparse attention, MoE | Preserve full-index versus reused-index identities. Initial dense FFN behavior depends on serving regime; it is not a generic dense-attention requirement. Top-k does not cap indexer history. |
| Kimi-K3 | 24 MLA and 69 KDA layers, latent MoE, separate draft | TP 8 gives 12 local heads from 96 global heads. Preserve irregular geometry, KDA verification/state slots, and native MLA versus granular-fallback identity. |
| MiniMax-M3.1 | MSA and MoE | [Official config](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/config.json) matches the registered M3 structural family. MXFP8 plus NVFP4-MoE precision and the initial dense-layer approximation still need exact matching. |
| DeepSeek-V4.1 | Compressed index/attention, window/reuse stages, mHC, Engram, MoE | [The consumer](../../src/aisimulate_core/sdk/models/deepseek_v41.py) is restricted to `nextn = 0`, DP 1, PP 1, CP 1 and no MegaMoE. Long-prefix data does not add unsupported speculative or parallel execution. |
| Qwen3.8-Flash-Next | Qwen4Exp, GDN, QSA compressed sparse attention, hyper-connections, PLE/ngram/conv state | [Official config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/236dfdf285828023ca3bcd3f37366c58a3469b13/config.json) identifies a different architecture from bundled Qwen3.8-Max. Its dispatch and extra operator contracts are missing at the checked revision. 256K is a future collection target. |
| Muse-Spark-Open | GQA/SWA/global attention and latent MoE | [Native serving source](https://github.com/sgl-project/sglang/blob/c1c287a42928495166e14dbcef31b9e1f1dac199/python/sglang/srt/models/muse_spark.py#L237) identifies a different family from Muse-Glimmer. The served config is gated; exact dimensions remain unknown. Do not substitute Glimmer geometry or claim qualified 1M dense coverage. |

Public model revisions establish available config identity, not the exact benchmark checkpoint revision when its native revision field is unset. More generic measurements cannot supply a missing model/operator contract.

## GPU admission and native qualification

The independent dense-context element guard is:

`2 × B_p × S × h_kv × d < 2,147,483,647`.

**2** counts K and V; `B_p × S` counts token positions; `h_kv × d` counts elements per token per array. At aggregate 1,048,576 it requires `h_kv × d < 1,024`. Some MHA shapes therefore remain excluded even if Blackwell memory is sufficient. Changing that guard requires native indexing evidence.

Memory admission is separate:

`M_est(shape, backend, dtype, pages, workspace) ≤ 0.90 × G`.

**0.90** is the existing 10% headroom policy; G is actual total device memory. A single-layer dense KV approximation is `2 × B_p × S × h_kv × d × b` bytes, where b is cache bytes/element. Real peak also includes Q/output copies, cache-layout copies, page rounding, metadata, and workspaces. Sparse logits can grow with query × history; recurrent state and compressed MLA use different formulas. The [memory guard](../attention_memory.py) is not applied to every family and cannot guarantee native peak safety.

A small GPU's failure must not remove feasible large-GPU shapes globally. As an illustrative estimate, TRT BF16 context `(B1,S1M,Q64,KV1,d256)` uses about 201 GiB in the current visible-buffer formula: it is excluded by an 80 GiB profile but admitted by a 267.689 GiB profile. Native workspace and kernel execution still need qualification. Previous 1M decode successes do not qualify 1M full-prefill cases.

Keep these constraints independent:

- [TRT standalone MLA](../trtllm/collect_mla.py): synthetic Scenario/RoPE capacity follows the cache-manager allocation; native decode-head/precision restrictions remain. Checkpoint limits apply independently to model-module collectors.
- [SGLang DSA](../sglang/collect_mla_module.py): context-only indexer aggregate-KV cap `2^25 = 33,554,432`, plus actual chunk/pool capacity. The context cap is not a decode filter and does not reject the 52 × 720,896 generation point.
- [vLLM DSA](../vllm/collect_mla_module.py): separate `B × (p + q) ≤ 1,048,576` full-history filter.
- [GDN/KDA](base_ops/gdn.yaml): convolution indexing and CUDA grid limits, independent of history coverage.
- [TRT A2A LL](../wideep/trtllm/collect_moe_a2a.py): top-k≤9 and allowed hidden geometry; a token increase does not add Qwen top-k10.
- [Dense native dispatch](../vllm/collect_attn.py): actual backend/page selection affects workspace. SGLang native MHA can retain two workspace allocations; TRT has packed QKV and doubled cache capacity. A guessed backend can admit the wrong memory plan.

Follow [layer permissions](../../.claude/rules/collector/layer_permissions.md) and [failure handling](../../.claude/rules/collector/failure_handling.md). Keep workload ranges in YAML; use only permitted, counted generation-time memory admission. Once queued, execute or raise a classified error. Ordinary kernel failures must not become new silent skip rules or fabricated rows. Distinguish a declaration, an admitted case, a measured result, and complete replay support.

## Disposition of all operation families

This table is the complete 36-family disposition. “Retain” is not proof that every geometry is covered. Registry activation, precision, model position, native dispatch, and actual data availability still apply. Analytical operators without measured tables do not acquire a collector budget simply by appearing in a model.

| Family | Budget | Status / qualification |
|---|---|---|
| Dense context | 64K/128K/512K/1M anchors and 1M aggregate cap | Implemented; [shared defaults](base_ops/attention_context.yaml), index/native constraints above |
| Dense generation | 16,777,216 aggregate total-token positions | Retain pending rank tuples; preserve feasible low-batch 1M anchors; [defaults](base_ops/attention_generation.yaml) |
| Encoder attention | Sequence 65,536; aggregate 131,072 | Retain; decoder history does not establish encoder demand; [defaults](base_ops/attention_encoder.yaml) |
| Standalone MLA context | 1M full-history aggregate | Implemented; synthetic TRT capacity follows its cache manager; [defaults](base_ops/mla.yaml) |
| Standalone MLA generation | 16M aggregate total-token positions | Retain pending tuples; synthetic TRT capacity now matches its cache manager |
| Generic MLA context | 1M full-history list and aggregate budget | Declares B1 tail; verify actual route, geometry, and execution; [defaults](base_ops/mla_module.yaml) |
| Generic MLA generation | vLLM/TRT 33,554,432; SG 37,486,592 aggregate past-token positions | Retain caps; SG history axis now reaches 1,048,575 |
| vLLM DSA context | 32K new-token aggregate; separate 1M aggregate full history | Implemented; requires matching local chunk and native indexer support |
| SGLang DSA context | q = 32K; aggregate 131,072 | Match native chunk; preserve independent model/index/pool bounds |
| TRT DSA context | q = 32K; aggregate 131,072 | Retain; qualify long-prefix logits/workspace |
| DSA generation | Generic MLA backend budgets above | Retain pending tuples; distinguish full-index and reused-index rows |
| MSA context | q = 32K; aggregate 131,072 | All three frameworks declare it. TRT Triton-reference aggregate 32K is a separate limit |
| MSA generation | 33,554,432 aggregate past-token positions | Retain; SG uses this cap rather than 37,486,592 |
| DSV4 module | vLLM/TRT default history 64K; SG model grid through 1M | Retain; no active V4-Pro basis for expansion; [defaults](base_ops/dsv4_attention.yaml) are separate from V4.1 |
| DSV4 sparse kernels | vLLM new-token budget 8K; SG runtime-derived chunk; aggregate full history 1M | Retain distinct paths; [model grid](models/DeepseekV4ForCausalLM_cases.yaml) does not alone set registered SG sparse shapes |
| DSV4 calibration | Owning module-derived q/p/B axes | No independent history budget; [calibration](../sglang/deepseekv4_sparse_modules.py) measures variant-qualified top-k distributions |
| DSV4.1 module | Conditional legal 1M history / 32K new-query manifests | Target after feature/backend eligibility; [default manifest](base_ops/dsv41_module.yaml) is short and unverified |
| GLM5.3-Flash attention | Regular 128K history; long class up to 1M at B1; 8K new-token budget | Retain; no active points justify expansion; [contract](base_ops/glm53flash_attention.yaml) |
| GLM5 sparse kernels | Exact DSA q/p/B axes and indexer identity | DSA query expansion implemented; 8,000,000 scoring elements trigger chunking, not a coverage ceiling; [implementation](../sglang/glm5_dsa_sparse_modules.py) |
| GEMM | 32,768 token rows | Retain unless actual mapped M exceeds it; verify N/K and precision; [defaults](base_ops/gemm.yaml) |
| Compute-scale / scale-matrix | 32,768 rows | Retain local M mapping; no automatic MoE DP expansion; [defaults](base_ops/compute_scale.yaml) |
| MLA BMM pre | Effective batch through 8,192 | Retain; consumer receives batch, not prefill-token total; [defaults](base_ops/mla_bmm_gen_pre.yaml) |
| MLA BMM post | Effective batch through 20,480 | Retain for the same reason; [defaults](base_ops/mla_bmm_gen_post.yaml) |
| mHC | New-token rows through 524,288 | Retain; operator variant/geometry remain separate; [defaults](base_ops/mhc.yaml) |
| GDN | Context q = 32K, B = 64; decode B = 1,024 | Retain pending actual request batches; no historical-token scan; [defaults](base_ops/gdn.yaml) |
| KDA | Same context/decode range; verify widths {2,3,4,5,6,8}, B ≤ 256 | Width expansion implemented; preserve state/geometry; [defaults](base_ops/kda.yaml) |
| Mamba2 | Context q = 32K, B = 64; decode B = 1,024 | Retain; no active-family basis for expansion; [defaults](base_ops/mamba2.yaml) |
| Ordinary MoE | 32K/64K rows; conditional 128K | 32K/64K implemented; TRT no longer silently shrinks; [defaults](base_ops/moe.yaml) |
| WideEP expert compute | TRT global 64K inherited; local 32K conditional | Preserve backend key mapping; SG local decode remains 128 until justified; [collectors](../wideep/) |
| Shared MoE A2A | HT 65,536 and LL 256 per-rank tokens | Retain; topology/geometry/receive buffers may limit execution first; [defaults](base_ops/moe_a2a.yaml) |
| Legacy SGLang DeepEP LL | Per-rank tokens through 1,024 | Retain; dynamically sized dispatch capacity and correct consumer-table identity; [collector](../wideep/sglang/collect_deepep_ll.py) |
| Legacy SGLang DeepEP normal | Per-rank tokens through 131,072 | Retain; its 1,024 correctness-check threshold does not cap performance collection; [collector](../wideep/sglang/collect_deepep_normal.py) |
| DSV4 MegaMoE | Local prefill 32K, decode 512 | Retain; no generic MoE DP multiplier; [collector](../sglang/collect_dsv4_megamoe.py) |
| NCCL | 1 GiB; conditional 2 GiB and 16-rank topology | 1 GiB default implemented; [collector](../network/collect_nccl.py) |
| Custom all-reduce | 536,870,912 elements, or 1 GiB at 2 bytes/element | Retain; units differ from NCCL CLI bytes; [collector](../network/collect_all_reduce.py) |
| TRT fixed-geometry all-to-all | 65,536 tokens at hidden width 7,168 / 256 experts / top-k 8 | Retain range; use model-correlated collection for other geometry; [collector](../network/slurm/collect_trtllm_alltoall.py) |

## Collection cost and qualification

**No additional full-sweep runtime has been measured for these targets.** Existing model-specific timings cannot establish all-model cost. Added shapes differ in geometry, precision, backend, warmup, graph capture, repetitions, and failure behavior.

Estimate incremental collection time as:

`ΔT = ΔT_setup_groups + Σ_{c∈new admitted cases} [t_warmup(c) + n_repeat(c) × t_run(c)] + ΔT_failures`.

- `c` is one distinct native case; `n_repeat(c)` is its timing repetition count.
- All time terms are seconds. `ΔT_setup_groups` charges additional worker/model-group initialization once per group, not once per inner shape.
- Deduplicate shared physical geometry and include old anchors made eligible by a larger budget. For example, B4 at the existing 262,144 anchor contributes cost even though the anchor itself is unchanged.

A structural census using the [actual getter test harness](../../tests/unit/collector/test_dense_attention_agentic.py), comparing defaults at commit `36a69a683c60081e61d6379f868b1c7257e1a41e` with the enabled defaults, produced the following result for a shared 1M cap plus 64K/128K/512K/1M anchors:

| Backend | Previous cases | Enabled cases | Additional cases | Optional 786,432 adds |
|---|---:|---:|---:|---:|
| vLLM | 71,858 | 80,270 | 8,412 | 414 |
| SGLang | 83,178 | 92,469 | 9,291 | 465 |
| TRT-LLM | 104,166 | 115,821 | 11,655 | 570 |

That is **29,358 additional cases before actual-device filtering**, plus 1,449 for the optional anchor. Reproduce the structural method by calling each actual getter body through the harness with the previous and enabled resolved context sweeps. Use its explicit SM103/backend/page mocks; retain original geometry/precision. Compare original and target case sets. CPU-only, 80 GiB, and 267.689 GiB estimate profiles, with and without the optional anchor, retained all old cases across 18 profiles. These are mock-based structural/memory estimates, not live native dispatch, GPU feasibility, or measured runtime. With the enabled defaults, added cases after estimated memory admission are **8,366 / 8,830 / 10,700** at 80 GiB and **8,412 / 9,291 / 11,637** at 267.689 GiB, ordered vLLM/SGLang/TRT-LLM. All old cases survive in each profile.

Relative to baseline `e6b1fe2`, the combined dense context/decode expansion adds **47,630 vLLM / 48,076 SGLang / 58,119 TRT-LLM** cases before device admission. At an **assumed** five steady-state seconds per added case, this is **66.15 / 66.77 / 80.72 GPU-hours**. Five seconds is an illustrative scenario, not a measured forecast or upper bound; module, MoE, recurrent and network work, queue time and cold setup are excluded. The new 29,358-case context increment alone contributes **40.78 GPU-hours** under that assumption.

The ordinary-MoE actual-getter census on mocked SM103 expands inner token cells while leaving setup-group counts unchanged:

| Backend | Groups | Previous cells | Enabled cells | Additional cells |
|---|---:|---:|---:|---:|
| vLLM | 3,135 | 84,645 | 90,915 | 6,270 |
| SGLang | 7,503 | 202,581 | 217,587 | 15,006 |
| TRT-LLM | 11,385 | 307,395 | 330,165 | 22,770 |

Each group changes from 27 to 29 token anchors: **7.41% more cells**, but the sum of token counts is **2.70×** the old sum. That ratio is a linear-work proxy, not elapsed runtime; routing, native chunking, autotuning and failures matter. Kimi KDA verification changes from **108 to 216 cells per backend**, with the 12 geometry/precision groups unchanged; all-phase KDA cells change from 600 to 708. The NCCL range adds one size per collective/dtype/topology combination: the eight-GPU local 4-collective × 2-dtype × 3-topology driver adds **24 measurements**, with native warmup/startup charged for each. None of these counts establishes GPU feasibility.

The module census compares the same baseline with enabled defaults without adding head/model/precision setup axes:

| Module scope | Previous inner cases | Enabled inner cases | Additional |
|---|---:|---:|---:|
| vLLM DSA context, one BF16 precision across all declared local heads, before model/memory constraints | 21,792 | 25,368 | 3,576 |
| SGLang DSA context worker, one head/precision/1M-model group, retaining index cap and using an unlimited test pool | 2,501 | 3,275 | 774 |
| SGLang MSA context, one head/precision/1M-model group, before memory admission | 3,500 | 3,584 | 84 |
| SGLang MSA generation, same group | 239 | 256 | 17 |
| SGLang GLM DSA generation, heads 8 / FP8 KV / BF16 projections, before memory admission | 250 | 267 | 17 |

The DSA worker comparison includes replacing the previous implicit 16K Blackwell chunk with manifest-sized capacity; it is not just a YAML length count. All old shapes survive these structural comparisons. GLM full/skip-indexer rows would increase from 500 to 534 for that selected manifest; existing 500-row measurements stay valid and unchanged. No fixed seconds-per-point estimate is established for these module grids. The native examples below show that long-prefix preparation and warmup can dominate; a 0.1–1 second assumption would understate the measured new SGLang context probes. Do not add counts across unlike scopes or count shared shapes twice.

Earlier measured campaigns provide limited context: 95 Qwen B300/vLLM prefill rows took **52.751 seconds** collector time (18 added anchors were not timed separately); 500 GLM B300/SGLang rows took **1.213 GPU-hours**, with about **30.1 minutes per group** attributed to its 96 added shapes, mostly history seeding. This is within-run attribution, not a controlled before/after comparison. These older campaigns do not qualify the newly enabled sizes.

For otherwise identical B1 full-prefill cases, increasing S from 262,144 to 1,048,576 changes ideal attention-pair work by:

`(1,048,576 / 262,144)^2 = 4^2 = 16`.

**4** is the length ratio; **16** is a compute-work ratio, not measured time. More generally dense work scales approximately as `B × h_q × d × S²`. A constant B×S token budget therefore does not bound compute cost. Real cached-prefix forwards may compute far fewer queries than full-prefill calibration. Targeted joint-shape collection and a future prefix-aware dense measurement contract can reduce unnecessary work; they do not justify removing feasible large shapes.

### B300 native qualification

The following bounded checks used signed collector commit `f7cd581b0005e9084ef51dc2e330d941fc00fdd4` on NVIDIA B300 SXM6 AC (SM103, 287,428,640,768 bytes). They execute unchanged native collector code. Dense and ordinary-MoE probes select actual generated cases; module probes execute explicit joint shapes with model-position validation. The separate generator census establishes grid membership and preservation, while native probes establish execution for only their selected shapes. Smoke rows and provenance remain local and are not packaged timing data. Native-call wall time includes collector initialization/warmup; it is distinct from the logged per-forward latency and from allocation time.

| Native collector | Qualified shape | Native-call wall time |
|---|---|---:|
| vLLM 0.31.0 dense context | B1, S1M, Q4/KV4, d128, BF16 | 10.780 s |
| SGLang 0.5.21 dense context | Same | 21.632 s |
| TRT-LLM 1.3.0rc29 dense context | Same | 25.017 s |
| vLLM / SGLang / TRT-LLM dense context | B4, S262,144, Q4/KV1, d128, BF16 | 2.917 / 5.985 / 7.066 s |
| TRT-LLM standalone MLA context | B1, S1M, one local head, BF16 | 8.617 s |
| TRT-LLM standalone MLA generation | B1, past 1,048,575, 16 local heads, BF16 / FP8 KV | 2.808 / 10.777 s |
| TRT-LLM ordinary MoE | GLM geometry H6,144 / I2,048 / E256 / top-k8, TP4/EP1, NVFP4, grouped 16K/32K/64K tokens | 27.833 s for all three counts |
| NCCL 2.30.7, four B300 GPUs | All-gather/reduce-scatter × half/int8 × 512 MiB/1 GiB gathered payload | 29.979 s for eight native measurements |

The nine dense/standalone probes passed in **6 min 17 s summed one-GPU allocation time**, including startup. The standalone 1M MLA context peak allocation was 5,813,575,168 bytes, about 40.2 MiB above its visible-buffer estimate; it remained comfortably within the device headroom. Admission estimates are not exact peak guarantees. TRT MoE produced all three rows (2.668/5.094/9.665 ms per forward), with 27.59 GiB peak allocated memory. Its full process took 38.263 s and its one-GPU allocation 96 s. The NCCL harness took 31.580 s including setup; its four-GPU allocation lasted 119 s. These are measured probe costs, not the incremental cost of a full sweep.

SGLang DSA prefill checks at prefix 262,144 and q32,768 passed at B1 and B4 (32K/128K aggregate new tokens). Within the loaded runner, point preparation/timing took **5.01/10.43 s**; logged forward latencies were **76.859/332.651 ms**. SGLang MSA at the same prefix/query and B1 took **33.89 s**, including **32.97 s eager warmup/JIT** and a **0.16 s timed loop** (15.936 ms per forward). vLLM 0.30.0 DSA passed B1×32K and B2×16K queries at that prefix, producing both full- and reused-indexer rows. Do not multiply cold process startup by every inner shape when estimating grouped collection.

SGLang generation also passed **DSA B32 at past 1,048,575** and **MSA B1 at past 917,504**. The DSA loaded-runner point took **75.43 s**, including **70.91 s history initialization**, while the timed decode was only **0.7897 ms**. The MSA point took **32.96 s**, including **31.89 s eager warmup/JIT**. All seven module probes passed and produced nine rows (seven full-index/module rows plus two vLLM reused-indexer rows). These costs explain why added long-history cells cannot be estimated from decode latency alone.

vLLM 0.31.0 Kimi KDA passed **B1/B8 × widths 3/5/6** at the TP8 geometry (12 heads × 128 dimensions), producing **12 rows**: native `causal_conv1d_update` plus `fused_recurrent_kda` for each of six cells. Native collection took **21.489 s** for all six cells; the full process took **23.310 s** and the one-GPU allocation **65 s**. This qualifies those widths and geometry, not every state layout or parallel topology.

SGLang 0.5.21 Kimi KDA attempted B1/B8 × widths 3/5/6 and produced **zero rows**: that pinned image lacks `sglang.kernels.ops.kimi_k3`. This is a runtime package/collector compatibility gap, not evidence that the new widths are infeasible. No replacement kernels or fabricated measurements were used. Full grids, other GPUs, numerical correctness, exact served checkpoints and end-to-end prediction accuracy remain unqualified; range expansion alone does not establish any of them.

Qualification order:

1. Resolve exact model, checkpoint, table, native backend, precision, local/global token mapping, verifier width, and parallel topology.
2. Capture per-rank forward records at representative low/high concurrency in warmup and profiling: rank/worker, request IDs, q/p or packed equivalent, verification width, cache/state reuse, and TP/DP/CP/DCP.
3. Map records through the correct consumer entry point. Identify missing joint tuples instead of crossing unrelated maxima.
4. Extend shared defaults and model geometry in their owning layers. Preserve lower coverage and measured data/provenance.
5. Run the actual generators; compare requested, admitted, filtered, measured, and failed shapes. Qualify native buffers and kernels on target GPUs, including low-batch long histories and large shapes that fit only larger GPUs. Do not interpret an 80 GiB failure as a global exclusion.
6. Measure setup, warmup, and execution separately, then estimate the admitted sweep and state remaining measurement gaps. Record the actual signed collector commit, native environment, dispatched kernel, and measurement method with new rows.

Coverage expansion does not require rewriting previously measured parquet or its provenance. It also does not establish timing accuracy, complete replay support, or plotting support; those require their own validation.
