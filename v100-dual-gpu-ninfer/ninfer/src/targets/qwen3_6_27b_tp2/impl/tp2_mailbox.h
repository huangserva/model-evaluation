#pragma once

// Small-message all-reduce for TP2 through mapped pinned host memory (this box has no working
// GPU peer access, so NCCL also goes through host memory; its LL protocol adds a flag per 8 bytes
// and costs ~26 us for a 40 KB decode residual against ~16 us here).
//
// Each CTA owns one 1024-element slice: it writes its partial to the rank's host slot, fences,
// raises its flag to the current step, waits for the peer's flag, and adds the peer's slice in
// rank order (rank 0 + rank 1 on both ranks, so the result is bit-identical). Per-CTA step counters
// live in device memory, so the kernel is replay-safe inside CUDA graphs; slots alternate by step
// parity, which is safe because a rank cannot reach step s+2 before the peer finished step s.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {

inline constexpr int kMailboxSlice     = 1024;             // bf16 elements per CTA
inline constexpr int kMailboxMaxBlocks = 64;               // => 64K elements, 128 KB
inline constexpr int kMailboxMaxElements = kMailboxSlice * kMailboxMaxBlocks;

// Row all-gather (vocabulary-sharded heads): each rank holds rows [rank*Nh, (rank+1)*Nh) of an
// [N = 2*Nh, T] output as a contiguous [Nh, T] block.
inline constexpr int kGatherChunk     = 8192;            // elements per CTA
inline constexpr int kGatherMaxBlocks = 128;
inline constexpr int kGatherMaxElements = kGatherChunk * kGatherMaxBlocks;  // 1M elements, 2 MB

struct alignas(128) MailboxShared {
    std::uint64_t magic;
    std::uint64_t pad0[15];
    std::uint64_t flag[2][kMailboxMaxBlocks][16];  // [rank][block], 128 B apart
    __nv_bfloat16 data[2][2][kMailboxMaxElements]; // [rank][parity][elements]
    std::uint64_t gflag[2][kGatherMaxBlocks][16];
    __nv_bfloat16 gdata[2][2][kGatherMaxElements];
};

void launch_mailbox_allreduce(__nv_bfloat16* x, int elements, int rank, MailboxShared* mailbox,
                              std::uint64_t* steps, cudaStream_t stream);

// out[N, T] (leading dimension N = 2 * local_rows) <- both ranks' local [local_rows, T] blocks.
void launch_mailbox_gather_rows(const __nv_bfloat16* local, int local_rows, int t,
                                __nv_bfloat16* out, int rank, MailboxShared* mailbox,
                                std::uint64_t* steps, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
