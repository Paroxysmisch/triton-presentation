---
theme: ./slidev-theme
title: Correctness-Driven Multi-Agent GPU Kernel Optimization
transition: morph
themeConfig:
  primary: '#6366f1'
  secondary: '#06b6d4'
  accent: '#f59e0b'
  author: Yash Shah (ys562)
---

# Correctness-Driven Multi-Agent GPU Kernel Optimization

Bridging the Optimization Gap with Multiple LLM Agents and Symbolic Verification

<style>
.slidev-layout h1 {
  font-size: clamp(40px, min(5vw, 8vh), 80px);
}
</style>

---

# The Challenge of GPU Kernel Development

Writing GPU kernels is **difficult and time consuming**, requiring:

<v-clicks>

- Detailed knowledge of the underlying **hardware**
- Deep understanding of the **algorithm** being implemented

</v-clicks>

<div v-click class="glass mt-8 p-4 text-center">

However, well-implemented kernels can yield speedups of **orders of magnitude**.

</div>

---
layout: section
---

# The Optimization Gap

<style>
.slidev-layout h1 {
  font-size: clamp(36px, min(5vw, 8vh), 72px);
}
</style>

---

# Compilers Are Not Enough

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

By the time a compiler matures enough to effectively utilize new functional units, the **next generation of silicon is already shipping**.

<div v-click class="glass mt-8 p-6 text-center text-xl">

We are left with a <strong>massive optimization gap</strong>.

</div>

---

# LLMs as a Solution

LLMs and agent-based systems promise a way to **fill this optimization gap** by:

<v-clicks>

- Leveraging existing knowledge learnt from training data
- Applying it to new kernels we are designing

</v-clicks>

<div v-click class="glass mt-6 p-4">

We are **not** expecting the LLM to come up with completely new algorithms, but simply follow optimization engineering patterns an experienced kernel implementor would -- this alone is enough to give **substantial speedups**.

</div>

---
layout: section
---

# The Correctness Problem

<style>
.slidev-layout h1 {
  font-size: clamp(36px, min(5vw, 8vh), 72px);
}
</style>

---

# Why Random Testing Falls Short

While there has been some work in agent-based kernel development, one standout issue has been **verifying correctness**.

<v-click>

Some works (e.g. TritonBench<Ref n="1" />) provide correctness guarantees via **random input testing**, however this does not cover edge cases.

</v-click>

<div v-click class="glass mt-4 p-4">

**Example:** If we pick random inputs for a softmax kernel between [0, 1], the kernel may seem to pass, but may have **numerical stability issues**.

</div>

<Footnote n="1">TritonBench: arxiv.org/abs/2502.14752</Footnote>

---

# Example: PyTorch Reference

<p class="micro-label">Reference Implementation</p>

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

<style>
.slidev-layout { --slidev-code-font-size: 10px; --slidev-code-line-height: 14px; }
</style>

---

# Example: Correct Triton Kernel

<p class="micro-label">Reference</p>

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

Key: subtracts `row_max` before `exp()` for numerical stability.

<style>
.slidev-layout { --slidev-code-font-size: 10px; --slidev-code-line-height: 14px; }
</style>

---

# Example: Buggy Kernel

<p class="micro-label">Bug Introduced</p>

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

<style>
.slidev-layout { --slidev-code-font-size: 10px; --slidev-code-line-height: 14px; }
</style>

---

# Why This Bug Escapes Random Testing

<div class="glass p-4">

With random inputs in **[0, 1]**, `exp(row)` won't overflow -- the bug **passes random testing**.

But with large values, `exp(row)` overflows to `inf`, producing **NaN** outputs.

</div>

<v-click>

<p class="micro-label mt-8">The core issue</p>

Random input testing samples from a **benign distribution**. Bugs that only manifest at boundary conditions or with adversarial inputs go undetected.

</v-click>

---
layout: section
---

# Our Approach

<style>
.slidev-layout h1 {
  font-size: clamp(36px, min(5vw, 8vh), 72px);
}
</style>

---
layout: statement
---

# An multi-agent system consisting of an optimizer and a correctness agent system

<style>
.slidev-layout h1 {
  font-size: clamp(28px, min(3.5vw, 6vh), 56px);
}
</style>

---

# A Multi-Agent System

<v-clicks>

- We introduce a multi-agent system with an **optimizer agent system** and a **correctness agent system**
- The correctness agent system tries to find a **counter-example** such that the optimized kernel produces an output that differs from a reference, unoptimized version

</v-clicks>

---
layout: section
---

# Correctness Agent System

<style>
.slidev-layout h1 {
  font-size: clamp(36px, min(5vw, 8vh), 72px);
}
</style>

---

# Correctness Agent: Overview

<v-clicks>

1. Transform the Triton kernel into **TTIR** and build a **Z3 symbolic execution model**
2. Take a **blockwise-compositional verification** approach to handle Z3 limitations (loops, collapsing tensor dimensions)
3. Each block corresponds to something we can create a **complete Z3 formula** for
4. Remaining lines are verified via the **LLM using full debugging tools** (e.g. Triton Interpret) -- the LLM produces evidence and reasoning output that humans can verify
5. If something suspicious is found, the correctness agent tries to **generate a counter-example**

</v-clicks>

---
layout: section
---

# Optimizer Agent System

<style>
.slidev-layout h1 {
  font-size: clamp(36px, min(5vw, 8vh), 72px);
}
</style>

---

# Optimizer Agent: Overview

<v-clicks>

1. **Benchmarking Agent** -- access to basic timing-based benchmarks and NVIDIA Nsight Compute metrics via performance counters; provides a performance baseline and optimization approaches
2. **Proposal Agent** -- makes optimization proposals based on benchmark data
3. **Optimizer Agent** -- implements optimizations as code diffs
4. **Correctness Agent** -- checks the optimization for correctness
5. **Proposal Agent** re-evaluates and the cycle repeats

</v-clicks>

---
layout: end
---

# Thank You

Yash Shah (ys562)

<style>
.slidev-layout h1 {
  font-size: clamp(36px, min(5vw, 8vh), 72px);
}
</style>
