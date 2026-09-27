# 两张 V100 跑 Qwen3.8-27B：NInfer 双卡张量并行

这个目录是一份可以直接编译的推理引擎源码。它让 Qwen3.8-27B（NVFP4 权重）在两张 **Tesla V100 32GB PCIe** 上合跑同一个请求，两张卡之间**没有 NVLink**，也**没有可用的 P2P 直连**。本仓库里的全部评测题目，也在这个配置上重新跑过一遍，结果见下文。

- `ninfer/`：完整源码快照，对应开发分支 `sm70-attn-opt` 的提交 `c9aaec6`，也就是 2026-09-26 起在生产上运行的版本。
- `patches/`：我们在上游基线之上的 11 个提交，用 `git format-patch` 导出，可以逐个查看具体改了什么。

## 来源与许可证

- 上游：[Neroued/ninfer](https://github.com/Neroued/ninfer)，Apache License 2.0。原始许可证保留在 `ninfer/LICENSE`。
- V100（sm_70）移植：[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)，基线提交 `b37d0dd`。
- 双卡部分是我们自己在上述基线上实现的。设计时参考了两个项目，但没有合并它们的代码：
  - [plus1998/NInfer-V100-Duo](https://github.com/plus1998/NInfer-V100-Duo)，同样是 V100 双卡，面向 NVLink 机器；
  - [ValerioDolci/ninfer-tp2](https://github.com/ValerioDolci/ninfer-tp2)，提出了没有 P2P 时，经主机锁页内存交换数据的思路。

## 在基线之上改了什么（11 个提交）

| 提交 | 日期 | 内容 |
|---|---|---|
| `07752db` | 09-25 | 重写 V100 上的短 token 注意力内核：8 个 warp 按 key 分工，每块 64 个 key，寄存器预取 |
| `2c7f876` | 09-25 | 同一内核里，int8 转 fp16 改用 PRMT 位运算。和上一个提交合起来，186K 上下文下单次调用从 2.79ms 降到 1.29ms |
| `7889625` | 09-25 | 重新调读 prompt 时的 D256 flash 注意力分块参数。186K prompt 的读取时间从 455 秒降到 379 秒 |
| `5b1fa54` | 09-26 | 注册每卡 12 个 q 头、2 个 kv 头的注意力形状，双卡时两张卡各分一半头，仍然走新内核 |
| `1bd3a5b` | 09-26 | Qwen3.8-27B 双卡张量并行：新增双卡模型定义，每层用 NCCL allreduce，并放进 CUDA Graph |
| `8fac611` | 09-26 | 双卡时长上下文的 KV 分段加长。186K 时每轮生成从约 44ms 降到约 40ms |
| `4a570e2` | 09-26 | 两张卡每轮互相核对已提交的 token；新增前置代理，负责把请求同时发给两张卡、出错时两卡一起重启 |
| `2779a37` | 09-26 | 两个行切分投影原来在内层循环里现场转换激活格式，只跑到约 350GB/s；改成先转成 fp16 再算，这两处耗时减少约三成（例如 NVFP4 5120×8704 从 80.9µs 降到 57.3µs），单卡同样受益 |
| `3f26f64` | 09-26 | 128KB 以下的卡间数据改走主机锁页内存（每次约 16µs，NCCL 要 26µs）；词表输出层和猜词头按词表拆到两张卡 |
| `9dbb10e` | 09-26 | NVFP4 残差相加并进矩阵乘的输出阶段；注意力分段合并从 29µs 降到 13µs；新增"猜词只看最近上下文"开关，默认关闭 |
| `c9aaec6` | 09-26 | 代理：每套部署用独立的主机内存交换文件 |

## 测试机器

- 2 × Tesla V100 32GB PCIe（PG500-216），PCIe 3.0，没有 NVLink。
- IOMMU 处在 DMA-FQ 模式。驱动报告 P2P 可用，实际打开后传 4KB 要 49ms，所以不走 P2P。代理启动时会自动设 `NCCL_P2P_DISABLE=1`。
- CUDA 12.8。NCCL 必须用 2.21，我们用的是 pip 包 `nvidia-nccl-cu12==2.21.5`。系统自带的 2.31 没有 sm_70 内核，allreduce 实际不会执行。

## 速度

测试 prompt 用编号的 Python 代码段或中文段落拼到指定长度，结尾加一句任务要求。所有档位都关闭思考，温度 0.7，MTP 猜 3 个 token，int8 KV。32K、128K、256K 三档在生产端口上测，每次生成 1024 个 token；表里两个数分别是第一次冷启动和同一 prompt 第二次命中缓存时的结果。8K 那一档是在开发环境里用同样的程序测的，只跑一次，生成 256 个 token。单位 tok/s。

| 上下文 | V100 双卡 代码 | V100 双卡 中文 | 首字等待（冷启动） | 每轮耗时 |
|---|---:|---:|---:|---:|
| 8K | 140.2 | 105.8 | 6.2 秒 | — |
| 32K | 141.2 / 134.7 | 94.5 / 95.0 | 29.4 秒 | 约 24.5ms |
| 128K | 113.9 / 109.1 | 81.1 / 83.7 | 152 秒 | 约 29ms |
| 256K（26 万 token） | 94.7 / 91.0 | 71.0 / 67.7 | 393 秒 | 约 35ms |

和单卡的对比（186K 代码 / 193K 中文）：

| 配置 | 代码 | 中文 | 测法 |
|---|---:|---:|---|
| 上游 V100 移植版，单卡 | 36.1 | 26.1 | 单次测量 |
| 本目录的程序，单卡运行 | 56.2 | 42.8 | 4 个随机种子的平均，每轮 56.9ms |
| 本目录的程序，双卡运行 | 101.9 | 73.4 | 4 个随机种子的平均，每轮 32.4ms |

作为参考，用同一份 prompt 测了一张 RTX 4090 48GB 上的单卡 NInfer（官方 groupwise-int 权重）：8K 代码 108.6、中文 112.2；32K 代码 112.5、中文 94.3；128K 代码 99.8、中文 77.6；256K 代码 80.9、中文 67.4。4090 读 prompt 快约 1.85 倍，256K 首字等待是 210 秒，V100 双卡是 393 秒。生成速度 V100 双卡各档都持平或略快。

## 质量

本仓库的两套题都在这个配置上重跑过。运行时和冻结协议的差异如下：MTP 猜 3 个 token（协议要求关闭）、int8 KV（协议是 q8_0）、引擎是 NInfer 双卡（协议是 llama.cpp）。题目哈希、提示词、对话模板、温度 0、输出上限 8192 都和协议一致，并且逐项核对过。

- **300 题基准**（`suites/core-300-v1`）：237/300，其中代码 124、困难推理 113，截断 29 次。和冻结的 Qwen3.8-27B Q4_K_M 基线 245/300 配对比较，差 −2.67 个百分点，McNemar p=0.20；和 Q8_0 基线 238/300 比，p=1.0。两边都没有检出显著差异。同一批题在上面那张 4090 上的单卡 NInfer 得 236/300。
- **十维快筛第二轮**：旧的 llama.cpp Q8_0 答案和本配置的答案匿名混在一起，由同一个裁判按同一份私有评分细则判分。本配置 78.6，旧版 68.4。差距主要来自旧版有 5 次在 16K 输出上限内没写出最终答案；其余 8 个维度两边持平（68.0 对 68.4）。
- **十维快筛第一轮**（关闭思考，跑 3 次）：78.8 / 67.6 / 77.6，中位数 77.6。

## 编译

其余依赖（FFmpeg 开发库、CMake 3.28+、Ninja 等）和上游 README 相同。

```bash
cmake -S ninfer -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DNINFER_NCCL_ROOT=/path/to/nccl-2.21   # 该目录下要有 include/ 和 lib/libnccl.so.2
cmake --build build --target ninfer-serve
```

`NINFER_NCCL_ROOT` 在 `ninfer/src/targets/qwen3_6_27b_tp2/CMakeLists.txt` 里有一个默认值，那是我们服务器上的路径，编译时请改成自己的。

## 切分权重

把官方的 Qwen3.8-27B nvfp4 权重文件（v2 容器）切成两份，每张卡各读一份，每份约 12.4GB。切分工具需要 PyTorch，要在 `ninfer/` 目录下运行。`--verify` 会检查两半能否逐位拼回原文件。

```bash
cd ninfer
python -m tools.tp2.shard_qwen38_27b /path/to/qwen3_8_27b_nvfp4.ninfer /path/to/out/qwen3_8_27b_nvfp4 --verify
# 生成 qwen3_8_27b_nvfp4.rank0.ninfer 和 qwen3_8_27b_nvfp4.rank1.ninfer
```

## 运行

由前置代理 `tools/tp2/tp2_proxy.py` 启动两个进程，每张卡一个。代理把同一个请求同时发给两个进程，只返回第 0 张卡的结果。两张卡每轮核对一次已提交的 token，对不上就报错。任何一个进程退出，或者卡住超过 `--stall-seconds`，代理就把两个进程一起重启，并给客户端返回 503。下面是我们生产用的参数：

```bash
python3 ninfer/tools/tp2/tp2_proxy.py \
  --listen 0.0.0.0:8080 --binary build/apps/ninfer-serve \
  --model-prefix /path/to/out/qwen3_8_27b_nvfp4 \
  --api-key-file /path/to/api-key --gpus 0,1 \
  --stall-seconds 180 --lockstep-timeout 300 \
  --lockstep-file /dev/shm/ninfer_tp2_lockstep --id-file /dev/shm/ninfer_tp2.id \
  --max-waiting 4 --pending-timeout 600 -- \
  --model-id qwen3.8-27b --max-context 262144 --kv-capacity auto \
  --max-concurrency 1 --prefill-chunk 2048 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --vision --preserve-thinking --seed 42
```

两张 32GB 卡在这组参数下每张约用 20.4GB，最长上下文 262,144 token。

可以用这些环境变量关掉对应的改动，方便对照：

| 变量 | 作用 |
|---|---|
| `NINFER_TP_MAILBOX=0` | 卡间数据全部改走 NCCL，不走主机锁页内存 |
| `NINFER_TP_SHARD_HEADS=0` | 词表输出层和猜词头不拆，两张卡各算一遍 |
| `NINFER_SM70_ATTN_V2=0` | 退回原来的短 token 注意力内核 |
| `NINFER_MTP_ATTN_WINDOW=W` | 猜词时只看最近 W 个 key。默认关闭，因为代码题的猜中率会从约 0.72 降到约 0.6 |

## 已知问题

- **两卡偶发不一致**：跑本仓库评测的约 330 个请求里出现过 1 次（`TP2 LOCKSTEP FAILURE: unit 1088 ... count 798/806`）。每轮核对拦下了它，代理把两个进程一起重启，约 23 秒恢复，没有输出错误内容。根因还没查清。
- **代理的取消处理有一个 bug**：客户端过早断开时，`client_watch` 线程可能抛出 `AttributeError: 'NoneType' object has no attribute 'shutdown'`，那次请求的取消可能没有传到两张卡。服务不会因此停下。
- **只在上面这台机器上验证过**：没有 P2P、没有 NVLink、正好两张卡。有 P2P 或 NVLink 的机器上走的路径还没测过。
