# XPU `topk_topp_sampler` kernel hangs at batch_size = 1

## Summary

On the Intel XPU vLLM fork (`release/sdv/2026_WW15`), running

```bash
pytest -v -s tests/v1/sample \
  --ignore=tests/v1/sample/test_logprobs.py \
  --ignore=tests/v1/sample/test_logprobs_e2e.py
```

hangs on every test in `test_sampling_params_e2e.py` that uses `n=1`
(`test_penalties`, `test_stop`, `test_stop_token_ids`, `test_detokenize_false`,
`test_bad_words`, `test_allowed_token_ids`, `test_seed`). `test_n_gt_1`
(`n=3`) passes.

The same suite on **upstream vLLM** (`mll/main` HEAD) does **not** hang —
upstream has none of the XPU‑specific sampler kernel code, so the failure
mode cannot be reproduced there.

## Root cause

The hang is in the SYCL kernel
`torch.ops._xpu_C.topk_topp_sampler` from the
[`vllm-xpu-kernels`](https://github.com/vllm-project/vllm-xpu-kernels)
package, defined in
`csrc/xpu/sampler/topk_topp_sampler_kernels.hpp`.

That kernel is wired into vLLM via `forward_xpu` in
`vllm/v1/sample/ops/topk_topp_sampler.py` (added in commit
`1a21d8ea4 use sample kernel`) and is enabled by default whenever
`VLLM_XPU_USE_SAMPLER_KERNEL=1` (the default).

### Why batch_size = 1 is special

The launcher uses **one SYCL workgroup per batch row**, with a fixed local
size of 512 work‑items:

```cpp
// csrc/xpu/sampler/topk_topp_sampler_kernels.hpp
sycl::range<1> local(group_size);   // 512
sycl::range<1> global(batch_size);  // = 1 for the failing tests
return sycl::nd_range<1>(global * local, local);
```

So at `batch_size == 1` the entire dispatch is a *single workgroup*. The
GPU has no other work it can interleave to hide a stall in that workgroup.

### Why the workgroup never finishes

Inside each workgroup, the top‑k threshold is found by a bisection over
the logit value range:

```cpp
do {
    pivot = (low + high) / 2;
    // count items >= pivot inside this workgroup
    pivot_count = sycl::reduce_over_group(group, pivot_count_local,
                                          sycl::plus<>());
    if (pivot_count == top_k_value)      break;
    else if (pivot_count <  top_k_value) high = pivot;
    else                                 low  = pivot;
} while ((high - low) > eps);   // eps = 1e-9, no iteration cap
```

The failing tests run `hmellor/tiny-random-LlamaForCausalLM` with random
weights. After the penalty / mask transforms the post‑processed logits
contain large runs of **identical or near‑identical floats** (and a lot of
`-inf` from masking). Two failure modes follow:

1. **Duplicate logits.** When more than `top_k_value` logits are equal,
   `pivot_count` jumps from `< top_k` to `> top_k` without ever hitting
   equality, and `(high - low)` never shrinks below `1e-9`. The loop is
   unbounded — no max‑iteration guard.
2. **`±inf` in `low`/`high`.** The initial `low` / `high` come from
   `reduce_over_group(min/max)` over raw logits which, after masking, can
   contain `-inf`. `(low + high) / 2` then produces `nan`, every later
   compare is false, and the bisection makes no progress.

Either way the single workgroup spins. Eventually the `xe` watchdog fires
(`guc_exec_queue_timedout_job` in `dmesg`) and kills the SYCL submit, but
PyTorch is stuck inside the synchronous wait for the kernel and the
EngineCore process appears hung.

For `n > 1` (`test_n_gt_1`, batch row count = 3) several workgroups run
concurrently, the GPU has enough independent work for at least one to
make progress and complete normally, and the test passes.

### py-spy stack at the hang

```
_to_list                       (vllm/v1/worker/gpu_model_runner.py:7112)
_bookkeeping_sync              (vllm/v1/worker/gpu_model_runner.py:3471)
sample_tokens                  (vllm/v1/worker/gpu_model_runner.py:4399)
sample_tokens                  (vllm/v1/worker/gpu_worker.py:745)
collective_rpc                 (vllm/v1/executor/uniproc_executor.py:80)
sample_tokens                  (vllm/v1/executor/uniproc_executor.py:120)
step                           (vllm/v1/engine/core.py:402)
```

The visible blocking call is `synchronize()` on the device, but the actual
work that never returns is the preceding `xpu_topk_topp_sampler` kernel.

### Relevant `dmesg` excerpt

```
xe ... guc_exec_queue_timedout_job ...
drm_sched_job_timedout+0x6d/0x110 [gpu_sched]
```

## Fix applied in this repo

Commit `14149a089 fix sampler` on
`mll/mll_fix_1083` / `release/sdv/2026_WW15`:

* `vllm/v1/sample/ops/topk_topp_sampler.py::forward_xpu` — fall back to
  `forward_native` whenever `logits.shape[0] <= 1`. The fast XPU kernel is
  still used for the common multi‑request case where it is safe.
* `vllm/config/vllm.py` — default `async_scheduling=False` on XPU when the
  user did not set it explicitly. Async scheduling triggers a separate
  cross‑stream `event.synchronize()` hang on XPU during fixture init that
  is unrelated to the sampler kernel issue but blocks the same tests.

### Verification

After the fix on `release/sdv/2026_WW15`:

```
pytest -v -s tests/v1/sample \
  --ignore=tests/v1/sample/test_logprobs.py \
  --ignore=tests/v1/sample/test_logprobs_e2e.py

=> 231 passed, 1 skipped, 1 failed in ~110s
```

The single remaining failure is the pre‑existing
`tests/v1/sample/test_topk_topp_sampler.py::TestTritonTopkTopp::test_topk_only[32000-1024]`
PyTorch‑vs‑Triton tie‑break mismatch on XPU (one element differs in the
sorted output). Upstream Intel CI already skips it via
`cb4ff07f8` / `40ee64c00`. It is unrelated to the hang.

## Suggested upstream kernel fix (`vllm-xpu-kernels`)

The right place to fix this for good is the kernel itself, in
`csrc/xpu/sampler/topk_topp_sampler_kernels.hpp`:

1. Bound the bisection loop with an explicit iteration cap (e.g. 64 — the
   bit width of `float` mantissa and exponent combined is plenty).
2. Sanitise `low` / `high`: replace `-inf` with `lowest_finite_float` (or
   the minimum finite logit) before computing the midpoint to avoid `nan`.
3. Treat `pivot_count` ties on duplicates explicitly — when
   `(high - low) <= eps` and `pivot_count != top_k_value`, deterministically
   pick the lower bound (already done at line 420‑422 but only after the
   loop has exited).
4. Optionally widen the launch so very small batches do not occupy a
   single hardware thread (e.g. duplicate the workgroup over an unused
   dimension), so the watchdog has something to time‑slice against.

Until the upstream kernel changes, the per‑call `if logits.shape[0] <= 1`
guard in `forward_xpu` is the minimal user‑space mitigation and matches
the conditions under which the bug actually fires.
