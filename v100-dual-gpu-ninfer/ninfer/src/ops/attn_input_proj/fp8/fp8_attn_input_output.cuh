#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kFp8AttnInputQueryRows = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyRows   = 1024;
inline constexpr std::int32_t kFp8AttnInputGateRows  = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyBegin  = kFp8AttnInputQueryRows;
inline constexpr std::int32_t kFp8AttnInputGateBegin = kFp8AttnInputKeyBegin + kFp8AttnInputKeyRows;
inline constexpr std::int32_t kFp8AttnInputValueBegin =
    kFp8AttnInputGateBegin + kFp8AttnInputGateRows;

static_assert((kFp8AttnInputQueryRows % 8) == 0);
static_assert((kFp8AttnInputKeyRows % 8) == 0);
static_assert((kFp8AttnInputGateRows % 8) == 0);

struct Fp8AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;
    // Row split of the fused [Q | K | Gate | V] projection. Defaults are the full model; a TP2
    // rank shard (7168 rows) carries 3072 / 512 (q = 3/7, k = 1/14 of the fused rows).
    std::int32_t query_rows = kFp8AttnInputQueryRows;
    std::int32_t key_rows   = kFp8AttnInputKeyRows;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        const std::int32_t key_begin   = query_rows;
        const std::int32_t gate_begin  = query_rows + key_rows;
        const std::int32_t value_begin = 2 * query_rows + key_rows;
        if (parent_row < key_begin) {
            return query + static_cast<std::int64_t>(token) * query_rows + parent_row;
        }
        if (parent_row < gate_begin) {
            return key + static_cast<std::int64_t>(token) * key_rows + parent_row - key_begin;
        }
        if (parent_row < value_begin) {
            return gate + static_cast<std::int64_t>(token) * query_rows + parent_row - gate_begin;
        }
        return value + static_cast<std::int64_t>(token) * key_rows + parent_row - value_begin;
    }

    __device__ __forceinline__ __nv_bfloat16* destination_fixed(std::int32_t parent_row,
                                                                std::int32_t token) const {
        if (parent_row < kFp8AttnInputKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kFp8AttnInputQueryRows + parent_row;
        }
        if (parent_row < kFp8AttnInputGateBegin) {
            return key + static_cast<std::int64_t>(token) * kFp8AttnInputKeyRows + parent_row -
                   kFp8AttnInputKeyBegin;
        }
        if (parent_row < kFp8AttnInputValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kFp8AttnInputGateRows + parent_row -
                   kFp8AttnInputGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kFp8AttnInputKeyRows + parent_row -
               kFp8AttnInputValueBegin;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        *destination(parent_row, token) = __float2bfloat16_rn(result);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

} // namespace ninfer::ops::detail
