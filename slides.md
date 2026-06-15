---
theme: seriph
background: https://cover.sli.dev
title: Correctness-Driven Multi-Agent GPU Kernel Development
info: |
  Correctness-Driven Multi-Agent GPU Kernel Development System
class: text-center
drawings:
  persist: false
transition: slide-left
comark: true
---

# Correctness-Driven Multi-Agent GPU Kernel Optimization

Yash Shah (ys562)

<div @click="$slidev.nav.next" class="mt-12 py-1" hover:bg="white op-10">
  Press Space for next page <carbon:arrow-right />
</div>

---
transition: fade-out
---

# The Challenge of GPU Kernel Development

Writing GPU kernels is **difficult and time consuming**, requiring:

<v-clicks>

- Detailed knowledge of the underlying **hardware**
- Deep understanding of the **algorithm** being implemented

</v-clicks>

<div v-click class="mt-8 p-4 rounded bg-green-500/10 border border-green-500/30">

However, well-implemented kernels can yield speedups of **orders of magnitude**.

</div>

---

# The Optimization Gap

Compilers can perform some optimizations effectively, but **macro-level optimizations** like Flash Attention remain out of scope.

<v-click>

The key ideas behind Flash Attention are simple and **portable across other kernels**:

</v-click>

<v-clicks>

1. **Fusing memory-bound operations** by tiling blocks into high-speed SRAM
2. **Computing local reductions incrementally** to avoid materializing giant intermediate matrices in HBM
3. **Utilizing recomputation** during the backward pass to drastically reduce memory footprint

</v-clicks>

---

# Novel Hardware Compounds the Problem

<div class="mt-4">

By the time a compiler matures enough to effectively utilize new functional units, the **next generation of silicon is already shipping**.

</div>

<div v-click class="mt-8 p-6 rounded bg-red-500/10 border border-red-500/30 text-center text-xl">

We are left with a <span v-mark.underline.red="2">massive optimization gap</span>.

</div>

---

# LLMs as a Solution

LLMs and agent-based systems promise a way to **fill this optimization gap** by:

<v-clicks>

- Leveraging existing knowledge learnt from training data
- Applying it to new kernels we are designing

</v-clicks>

<div v-click class="mt-6 p-4 rounded bg-blue-500/10 border border-blue-500/30">

We are **not** expecting the LLM to come up with completely new algorithms, but simply follow optimization engineering patterns an experienced kernel implementor would -- this alone is enough to give **substantial speedups**.

</div>

---

# The Correctness Problem

While there has been some work in agent-based kernel development, one standout issue has been **verifying correctness**.

<v-click>

Some works (e.g. TritonBench) provide correctness guarantees via **random input testing**, however this does not cover edge cases.

</v-click>

<div v-click class="mt-4 p-4 rounded bg-yellow-500/10 border border-yellow-500/30">

**Example:** If we pick random inputs for a softmax kernel between [0, 1], the kernel may seem to pass, but may have **numerical stability issues**.

</div>

---

# Example: PyTorch Reference

A correct softmax with numerical stability:

```python
import torch

def softmax(x: torch.Tensor) -> torch.Tensor:
    # Subtract max for numerical stability
    x_max = x.max(dim=-1, keepdim=True).values
    x_shifted = x - x_max
    # Exponentiate
    exp_x = torch.exp(x_shifted)
    # Normalize
    sum_exp = exp_x.sum(dim=-1, keepdim=True)
    return exp_x / sum_exp
```

---

# Example: Correct Triton Kernel

```python {all|12-14|all}
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    row_shifted = row - row_max
    numerator = tl.exp(row_shifted)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

<div class="mt-2 text-sm opacity-75">

Key: subtracts `row_max` before `exp()` for numerical stability.

</div>

---

# Example: Buggy Kernel

```python {all|13-14|all}
@triton.jit
def softmax_kernel_optimized(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    # BUG: forgot to subtract row_max for numerical stability
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

<div v-click class="mt-2 p-3 rounded bg-red-500/10 border border-red-500/30 text-sm">

With random inputs in [0, 1], `exp(row)` won't overflow -- the bug **passes random testing**. But with large values, `exp(row)` overflows to `inf`, producing `NaN` outputs.

</div>

---

# Our Approach: Adversarial Multi-Agent System

<v-clicks>

- We introduce a multi-agent system with an **optimizer agent system** and a **correctness agent system** that are <span v-mark.circle.red="3">adversarial</span>
- The correctness agent system tries to find a **counter-example** such that the optimized kernel produces an output that differs from a reference, unoptimized version

</v-clicks>

<div v-click class="mt-6">

```mermaid {scale: 0.7}
graph LR
    O[Optimizer Agent System] -->|optimized kernel| C[Correctness Agent System]
    C -->|counter-example / approval| O
    style O fill:#4a9eff,color:#fff
    style C fill:#ff4a4a,color:#fff
```

</div>

---

# Correctness Agent System

<v-clicks>

1. Transform the Triton kernel into **TTIR** and build a **Z3 symbolic execution model**
2. Take a **blockwise-compositional verification** approach to handle Z3 limitations (loops, collapsing tensor dimensions)
3. Each block corresponds to something we can create a **complete Z3 formula** for
4. Remaining lines are verified via the **LLM using full debugging tools** (e.g. Triton Interpret) -- the LLM produces evidence and reasoning output that humans can verify
5. If something suspicious is found, the correctness agent tries to **generate a counter-example**

</v-clicks>

---

# Optimizer Agent System

<v-clicks>

1. **Benchmarking Agent** -- access to basic timing-based benchmarks and NVIDIA Nsight Compute metrics via performance counters; provides a performance baseline and optimization approaches
2. **Proposal Agent** -- makes optimization proposals based on benchmark data
3. **Optimizer Agent** -- implements optimizations as code diffs
4. **Correctness Agent** -- checks the optimization for correctness
5. **Proposal Agent** re-evaluates and the cycle repeats

</v-clicks>


---
layout: center
class: text-center
---

# Thank You

Yash Shah (ys562)

