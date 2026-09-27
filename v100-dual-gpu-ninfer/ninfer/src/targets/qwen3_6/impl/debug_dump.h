#pragma once
// Debug only: NINFER_DUMP_DIR=<dir> writes the last column of `x` to <dir>/<name>.bin (raw BF16)
// the first time each name is seen outside stream capture. Used for TP2 bring-up comparisons.
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::debug {
template <class TensorT>
inline void dump_last_column_once(const TensorT& x, const char* name, cudaStream_t s) {
    static const char* dir = std::getenv("NINFER_DUMP_DIR");
    static std::set<std::string> done;
    if (dir == nullptr) { return; }
    const std::string key = std::string(name) + "_T" + std::to_string(static_cast<long long>(x.ne[1]));
    if (done.count(key) != 0) { return; }
    cudaStreamCaptureStatus cap{};
    if (cudaStreamIsCapturing(s, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) { return; }
    done.insert(key);
    const std::size_t bytes = static_cast<std::size_t>(x.ne[0]) * 2;
    const auto* src = static_cast<const unsigned char*>(x.data) +
                      (static_cast<std::size_t>(x.ne[1]) - 1) * bytes;
    std::vector<unsigned char> host(bytes);
    cudaStreamSynchronize(s);
    cudaMemcpy(host.data(), src, bytes, cudaMemcpyDeviceToHost);
    char path[512];
    std::snprintf(path, sizeof(path), "%s/%s_T%lld.bin", dir, name, static_cast<long long>(x.ne[1]));
    if (FILE* f = std::fopen(path, "wb")) { std::fwrite(host.data(), 1, bytes, f); std::fclose(f); }
}
} // namespace ninfer::debug
