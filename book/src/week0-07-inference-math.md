# Week 0 Day 7: Inference Systems Math

Weeks 2 and 3 are about making inference fast, and every optimization there starts from a back-of-the-envelope
estimate: how many bytes the weights take, how big the KV cache grows, whether a step is limited by compute or by
memory bandwidth, and how much quantization saves. Today you will write those estimates as code, implement the group-wise
4-bit format that the Qwen3 MLX checkpoints use, and tile a matrix multiply the way a GPU kernel does.

```
src/tiny_llm_week0/systems.py
```

```bash
pdm run test --week 0 --day 7
```

**📚 Readings**

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [Roofline model](https://en.wikipedia.org/wiki/Roofline_model)
- [How to Scale Your Model: All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/)
- [Apple: Metal compute programming](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)
  and [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
  (threads, threadgroups, and SIMD groups; read before Week 2 Day 2)

## Task 1: Count Parameters

`ModelConfig` and `QWEN3_0_6B` (copied from the model's `config.json`) are supplied. Implement
`param_count(cfg, include_embeddings=True)` for a Qwen3-style dense decoder. Each layer has:

- query and output projections, `hidden x (num_heads * head_dim)` each;
- key and value projections, `hidden x (num_kv_heads * head_dim)` each;
- a SwiGLU MLP with three `hidden x intermediate` matrices;
- two RMSNorm weight vectors of size `hidden`, plus two of size `head_dim` when `qk_norm` is set.

Add one final RMSNorm, the embedding table (`vocab x hidden`), and a separate output projection of the same size only
when `tie_word_embeddings` is false. Your answer should match Qwen3-0.6B's published size: about 0.6B parameters, 0.44B
of them outside the embeddings.

## Task 2: Size the KV Cache

Implement `kv_cache_bytes(cfg, seq_len, batch_size, bytes_per_element)`: keys and values, for every layer, KV head,
head dimension, token, and request.

Qwen3-0.6B needs 112 KiB per token in BF16, so a 4,096-token context costs 448 MiB per request, more than the
whole 4-bit model's weights. Grouped-query attention (8 KV heads serving 16 query heads) halves this. Week 3's paged KV cache
exists because this memory, not compute, limits how many requests fit at once.

## Task 3: Rooflines and Decode Speed

- `roofline_time(flops, bytes_moved, peak_flops, bandwidth)`: a kernel takes at least `max(flops / peak, bytes / bandwidth)`.
  Return that time and whether memory is the limit.
- `decode_tokens_per_second(weight_bytes, kv_bytes, bandwidth)`: generating one token for one request reads every weight
  and the whole KV cache once, so the speed is at most `bandwidth / (weight_bytes + kv_bytes)`.

Try it for your own Mac. Look up its unified memory bandwidth (for example, about 100 GB/s on a base M2, 120 GB/s on a
base M4, and several hundred GB/s on Pro and Max chips), estimate the weight bytes of the 4-bit 0.6B model, and write
down the ceiling. In Week 2 you will measure real decode throughput and can compare it with this number.

## Task 4: Group-Wise Quantization

The Qwen3 MLX checkpoints store each projection as 4-bit codes plus a scale and a bias for every group of 128 consecutive
weights in a row. Implement:

- `quantize_groupwise(w, group_size, bits)`: for each group, `bias = min`, `scale = (max - min) / (2^bits - 1)`, and
  `q = round((w - bias) / scale)` clipped to `[0, 2^bits - 1]`, stored as `uint8`. Use a scale of `1` for constant
  groups. Raise `ValueError` if the columns do not split into whole groups.
- `dequantize_groupwise(q, scales, biases, group_size)`: `w ≈ q * scale + bias`.
- `pack_int4(q)` and `unpack_int4(packed)`: eight 4-bit codes per `uint32`, with element 0 in the lowest four bits,
  matching the layout described in [Week 2 Day 2](./week2-02-quantize-model.md).

The reconstruction error of each weight is at most half a quantization step. Count the bytes: 4 bits per weight plus
two 16-bit values per 128 weights is about 4.25 bits per weight, almost 4x smaller than BF16. For a memory-bound decode,
that is close to a 4x speedup, which is why Week 2 keeps weights packed instead of dequantizing them up front.

## Task 5: Tiled Matrix Multiplication

Implement `tiled_matmul(a, b, tile)`. Loop over output tiles of size `tile x tile`. For each output tile, keep an
accumulator and add the product of one `tile`-wide slice of `a` and `b` at a time along `K`. Handle edges where the
sizes are not multiples of `tile`.

On a GPU, each output tile maps to a threadgroup that loads its input tiles into fast on-chip memory once and reuses them
many times, which raises arithmetic intensity. Week 2's SIMD matrix prefill and tiled attention kernels follow this
structure, and FlashAttention in Week 3 applies the same idea to softmax with Day 1's running maximum.

{{#include copyright.md}}
