#include "targets/qwen3_6_27b_tp2/impl/tp2_mailbox.h"

#include "core/device.h"

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {
namespace {

__global__ void mailbox_allreduce_kernel(__nv_bfloat16* x, int n, int rank, MailboxShared* mb,
                                         std::uint64_t* steps) {
    const int blk            = static_cast<int>(blockIdx.x);
    const std::uint64_t step = steps[blk] + 1;
    const int parity         = static_cast<int>(step & 1);
    const int peer           = 1 - rank;
    const int base           = blk * kMailboxSlice;
    const int end            = min(n, base + kMailboxSlice);
    for (int i = base + static_cast<int>(threadIdx.x) * 8; i < end; i += blockDim.x * 8) {
        *reinterpret_cast<uint4*>(&mb->data[rank][parity][i]) =
            *reinterpret_cast<const uint4*>(&x[i]);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile std::uint64_t*>(&mb->flag[rank][blk][0]) = step;
        // A dead peer must not leave this kernel spinning forever (the process could not exit and
        // the supervisor's restart would stall): give up after ~60 s and fault the context.
        const long long start = clock64();
        while (*reinterpret_cast<volatile std::uint64_t*>(&mb->flag[peer][blk][0]) < step) {
            if (clock64() - start > 83'000'000'000LL) { __trap(); }
        }
    }
    __syncthreads();
    __threadfence_system();
    for (int i = base + static_cast<int>(threadIdx.x) * 8; i < end; i += blockDim.x * 8) {
        uint4 mine        = *reinterpret_cast<const uint4*>(&x[i]);
        const uint4 other = __ldcv(reinterpret_cast<const uint4*>(&mb->data[peer][parity][i]));
        auto* a           = reinterpret_cast<__nv_bfloat162*>(&mine);
        const auto* b     = reinterpret_cast<const __nv_bfloat162*>(&other);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 fa = __bfloat1622float2(a[j]);
            const float2 fb = __bfloat1622float2(b[j]);
            // rank 0's value first on both ranks: identical rounding everywhere
            const float2 s  = rank == 0 ? make_float2(fa.x + fb.x, fa.y + fb.y)
                                        : make_float2(fb.x + fa.x, fb.y + fa.y);
            a[j] = __float22bfloat162_rn(s);
        }
        *reinterpret_cast<uint4*>(&x[i]) = mine;
    }
    if (threadIdx.x == 0) { steps[blk] = step; }
}

__global__ void mailbox_gather_rows_kernel(const __nv_bfloat16* local, int nh, int t,
                                           __nv_bfloat16* out, int rank, MailboxShared* mb,
                                           std::uint64_t* steps) {
    const int blk            = static_cast<int>(blockIdx.x);
    const std::uint64_t step = steps[blk] + 1;
    const int parity         = static_cast<int>(step & 1);
    const int peer           = 1 - rank;
    const int total          = nh * t;
    const int base           = blk * kGatherChunk;
    const int end            = min(total, base + kGatherChunk);
    const int n              = 2 * nh;
    // publish this rank's chunk and copy it to its own rows of out
    for (int e = base + static_cast<int>(threadIdx.x) * 8; e < end; e += blockDim.x * 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(&local[e]);
        *reinterpret_cast<uint4*>(&mb->gdata[rank][parity][e]) = v;
        const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int idx = e + j;
            const int col = idx / nh;
            const int row = idx - col * nh;
            out[static_cast<std::int64_t>(col) * n + rank * nh + row] = pv[j];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile std::uint64_t*>(&mb->gflag[rank][blk][0]) = step;
        const long long start = clock64();
        while (*reinterpret_cast<volatile std::uint64_t*>(&mb->gflag[peer][blk][0]) < step) {
            if (clock64() - start > 83'000'000'000LL) { __trap(); }
        }
    }
    __syncthreads();
    __threadfence_system();
    for (int e = base + static_cast<int>(threadIdx.x) * 8; e < end; e += blockDim.x * 8) {
        const uint4 v = __ldcv(reinterpret_cast<const uint4*>(&mb->gdata[peer][parity][e]));
        const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int idx = e + j;
            const int col = idx / nh;
            const int row = idx - col * nh;
            out[static_cast<std::int64_t>(col) * n + peer * nh + row] = pv[j];
        }
    }
    if (threadIdx.x == 0) { steps[blk] = step; }
}

} // namespace

void launch_mailbox_gather_rows(const __nv_bfloat16* local, int local_rows, int t,
                                __nv_bfloat16* out, int rank, MailboxShared* mailbox,
                                std::uint64_t* steps, cudaStream_t stream) {
    const int total  = local_rows * t;
    const int blocks = (total + kGatherChunk - 1) / kGatherChunk;
    mailbox_gather_rows_kernel<<<blocks, 256, 0, stream>>>(local, local_rows, t, out, rank, mailbox,
                                                            steps);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mailbox_allreduce(__nv_bfloat16* x, int elements, int rank, MailboxShared* mailbox,
                              std::uint64_t* steps, cudaStream_t stream) {
    const int blocks = (elements + kMailboxSlice - 1) / kMailboxSlice;
    mailbox_allreduce_kernel<<<blocks, 64, 0, stream>>>(x, elements, rank, mailbox, steps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
