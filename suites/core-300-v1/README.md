# 每模型 300 题代码与困难推理基准

这是一套**已退役的历史实测快照**。名称里的“300”表示每个模型配置运行 300 道相同题目；“v1”表示第一版公开快照。它适合复现实验与审计历史结论，不应冒充仍未公开的新题泛化测试。

## 题目构成

| 类型 | 上游数据集 | 每模型题数 | 判定 |
|---|---|---:|---|
| Python 代码 | HumanEval+ | 75 | EvalPlus 固定版本执行测试 |
| Python 代码 | MBPP+ | 75 | EvalPlus 固定版本执行测试 |
| 多领域困难选择题推理 | MMLU-Pro | 150 | `FINAL=<选项字母>` 精确匹配 |
| **合计** |  | **300** |  |

“每模型 300 题”是核心口径。原始两种量化配置均完成 300/300，后续 ApoDEx 与 Qwen Flash-Next 也各完成同一批 300/300；[结果表](./results.md)只报告这些完整运行。

HumanEval+ 与 MBPP+ 是 Python 代码生成题库；MMLU-Pro 是覆盖多学科的困难选择题题库；EvalPlus 是执行代码并检查其正确性的开源评测器。

本快照只收录已经完整跑完的 300 题核心集。历史实验中的其他未完成探索不在本目录中，也不参与这里的模型比较。

## 公开内容

- [`manifest.jsonl`](./manifest.jsonl)：300 条确切题目身份、上游行号、题目哈希与提示词哈希。JSONL 是“一行一个 JSON 对象”的文本格式。
- [`sources.json`](./sources.json)：数据集、文件与评测器的固定版本和文件哈希。
- [`scripts/rebuild_prompts.py`](./scripts/rebuild_prompts.py)：从固定上游版本重建模型当时看到的 300 条提示词，并逐条核对哈希。
- [`protocol.md`](./protocol.md)：共同运行边界、评分和统计规则。
- [`results.md`](./results.md)：四个完整模型配置的历史结果。

公开清单不含正确答案、评分器内部字段、历史盲测顺序、模型输出、本机路径或访问凭据。

## 快速校验

只校验仓库中已提交的 300 条清单，不需要联网或第三方依赖：

```bash
cd suites/core-300-v1
python3 scripts/verify_publication.py
```

预期输出包含：总数 300、HumanEval+ 75、MBPP+ 75、MMLU-Pro 150，以及冻结任务集合哈希：

```text
b0beacae2973ff00a30af0b36a699d1e12bfaf29d386aad8b599ed252f2344d6
```

## 重建模型实际看到的题面

重建需要联网下载固定的上游快照，并安装 `pyarrow`（用于读取 Parquet 数据文件）：

```bash
cd suites/core-300-v1
python3 -m pip install -r requirements.txt
python3 scripts/rebuild_prompts.py --output /tmp/core-300-prompts.jsonl
```

脚本只有在 3 个数据集的 5 个上游制品哈希和 300 条提示词哈希全部匹配时才会成功。5 个制品是 3 个 Parquet 数据文件和 2 个 EvalPlus 发布文件；Parquet 是用于保存表格数据的列式文件格式。输出只含题号与题面，不含答案。

依赖文件固定的是 `pyarrow 18.1.0`；推荐使用 Python 3.9–3.13，因为该版本的官方包元数据只覆盖这些 Python 版本。

## 历史冻结身份

- 完整内部冻结清单的 SHA-256：`5ff786c944a8bb151531082f2b197f9cae45e5e3ebf71848c0e27cb717d873f9`
- 300 个冻结任务哈希按清单顺序组成的集合哈希：`b0beacae2973ff00a30af0b36a699d1e12bfaf29d386aad8b599ed252f2344d6`
- 不含答案的公开清单 SHA-256：`703a6e5a9073f2b434a175eeefbb48a7ec4024027c6cf9e7f29846d3c6d74eec`
- 协议版本：`qwen38-q8-q4-v1`

SHA-256 是用于验证文件或数据是否逐字节一致的 256-bit 摘要，通常写成 64 个十六进制字符；这里用于证明公开清单对应当时真正运行的那批题。
