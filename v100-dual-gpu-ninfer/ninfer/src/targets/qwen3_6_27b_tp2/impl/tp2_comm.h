#pragma once

// Tensor-parallel (TP2) collectives for the qwen3_6_27b_tp2 target.
//
// Each rank is its own ninfer process pinned to one GPU (CUDA_VISIBLE_DEVICES), loading its rank
// artifact (tools/tp2/shard_qwen38_27b.py). A front proxy sends identical requests to both, so the
// two engines execute the same op sequence; the collectives below are the only coupling.
//
// Environment: NINFER_TP_RANK (0 or 1) and NINFER_TP_ID_FILE (rank 0 writes the NCCL unique id
// there, rank 1 reads it; the launcher deletes it before starting both ranks).

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {

// Creates the communicator and warms every message size used by decode, outside graph capture.
void init();

[[nodiscard]] int rank();

// residual (BF16) <- sum over ranks, in place.
void allreduce(Tensor& residual, cudaStream_t stream);

// residual <- residual + sum over ranks of partial: rank 0 adds its partial, rank 1 replaces the
// residual with its partial, then one all-reduce.
void combine_partial(const Tensor& partial, Tensor& residual, cudaStream_t stream);

// Vocabulary-sharded output head: this rank computes rows [rank*N/2, (rank+1)*N/2) of
// hidden x head^T and both ranks end with the full [N, T] out (bit-identical on both ranks).
// Returns false when the head/shape is not shardable here; the caller then runs it whole.
bool head_linear(const Tensor& hidden, const Weight& head, Tensor& out, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
