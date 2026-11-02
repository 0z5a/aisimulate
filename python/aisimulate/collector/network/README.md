# Network Collectors

This folder contains collective communication and network-facing collector
scripts. These collectors are intentionally separate from framework op
registries because their output feeds shared communication tables instead of a
single framework backend registry.

- `collect_comm.sh`: local NCCL/oneCCL plus custom allreduce collection driver.
- `collect_nccl.py`: local NCCL collective benchmark wrapper.
- `collect_oneccl_xpu.py`: local oneCCL XPU collective benchmark wrapper.
- `collect_all_reduce.py`: local custom allreduce benchmark wrapper.
- `slurm/`: multi-node Slurm communication collectors and post-processing.

The standalone scripts keep their staging output names (`nccl_perf.txt`,
`oneccl_perf.txt`, `custom_allreduce_perf.txt`, and
`trtllm_alltoall_perf.txt`). Collector finalization converts accepted output
to parquet under
`src/aisimulate_core/systems/data/<system>/comm/<backend>/<version>/`.

NCCL sweeps include payloads from 512 bytes through 1 GiB by default, using
`--range 512,1073741825,2` (exclusive upper endpoint). This brackets the
768 MiB gathered activation envelope described in
[agentic workload coverage](../cases/agentic_coverage.md). The CLI range is
bytes; persisted `message_size` is elements after dtype conversion. For
all-gather and reduce-scatter, compare the gathered buffer size rather than
one rank's contribution. The local driver retains its 2/4/8-GPU groups;
2 GiB and 16-rank collection still require separate qualification.
