# 每模型 300 题代码与困难推理基准

这是一套**已退役的历史实测快照**。名称里的“300”表示每个模型配置运行 300 道相同题目；“v1”表示第一版公开快照。它适合复现实验与审计历史结论，不应冒充仍未公开的新题泛化测试。

## 题目构成

| 类型 | 上游数据集 | 每模型题数 | 判定 |
|---|---|---:|---|
| Python 函数生成题 | HumanEval Plus（HumanEval 增强版，HumanEval+）；加号表示增加了更强测试 | 75 | 开源代码执行评测器（EvalPlus）固定版本 |
| Python 基础编程题 | Mostly Basic Python Problems Plus（基础 Python 编程题增强版，MBPP+）；加号表示增加了更强测试 | 75 | 开源代码执行评测器（EvalPlus）固定版本 |
| 多领域困难选择题推理 | Massive Multitask Language Understanding Pro（多学科语言理解专业难度版，MMLU-Pro）；Pro 表示更困难的增强版本 | 150 | 最终选项字母精确匹配 |
| **合计** |  | **300** |  |

“每模型 300 题”是核心口径。四个模型配置采用冻结的本地推理协议并各自完成 300/300。[结果表](./results.md)另单列一个腾讯 WorkBuddy 渠道扩展对照。该入口使用 WorkBuddy 显示的预览模型别名 Hy4 Preview；名称中数字 4 的含义平台没有披露，不能把它当作已验证的固定版本号。它完成同一批题、逐字相同的用户提示词和同一评分，但平台系统上下文及部分冻结推理参数不同，因此不并入严格协议主表。

HumanEval+ 与 MBPP+ 是 Python 代码生成题库；MMLU-Pro 是覆盖多学科的困难选择题题库；EvalPlus 是执行代码并检查其正确性的开源评测器。

本快照只收录已经完整跑完的 300 题核心集。历史实验中的其他未完成探索不在本目录中，也不参与这里的模型比较。

## 公开内容

- 逐行结构化文本（JavaScript Object Notation Lines，简称 JSONL；每行一个 JSON 对象）的公开清单：[`manifest.jsonl`](./manifest.jsonl)，含 300 条确切题目身份、上游行号、题目哈希与用户提示词哈希。
- [`sources.json`](./sources.json)：数据集、文件与评测器的固定版本和文件哈希。
- [`scripts/rebuild_prompts.py`](./scripts/rebuild_prompts.py)：从固定上游版本重建模型当时看到的 300 条提示词，并逐条核对哈希。
- [`protocol.md`](./protocol.md)：共同运行边界、评分和统计规则。
- [`results.md`](./results.md)：四个严格协议配置的历史结果，以及一个非严格同协议的 WorkBuddy 渠道扩展对照。
- [`hy4-workbuddy-2026-08-29.md`](./hy4-workbuddy-2026-08-29.md)：Hy4 Preview 的 300 题公开结果卡、配对统计和协议差异。

公开清单不含正确答案、评分器内部字段、历史盲测顺序、模型输出、本机路径或访问凭据。

## 快速校验

只校验仓库中已提交的 300 条清单，不需要联网或第三方依赖。下面的 `cd` 命令进入套件目录，`python3` 命令使用 Python 3 运行校验脚本：

```bash
cd suites/core-300-v1
python3 scripts/verify_publication.py
```

预期输出包含：总数 300、HumanEval+ 75、MBPP+ 75、MMLU-Pro 150，以及冻结任务集合哈希：

```text
b0beacae2973ff00a30af0b36a699d1e12bfaf29d386aad8b599ed252f2344d6
```

## 重建模型实际看到的题面

重建需要联网下载固定的上游快照，并安装用于读取 Parquet 列式数据文件的 Python 库（PyArrow）。下面第一条 Python 命令中的 `-m pip` 表示调用 Python 包安装器，`-r requirements.txt` 表示按依赖清单安装；第二条命令中的 `--output` 指定重建题面的输出文件：

```bash
cd suites/core-300-v1
python3 -m pip install -r requirements.txt
python3 scripts/rebuild_prompts.py --output core-300-prompts.jsonl
```

脚本只有在 3 个数据集的 5 个上游制品哈希和 300 条提示词哈希全部匹配时才会成功。5 个制品是 3 个 Parquet 数据文件和 2 个 EvalPlus 发布文件；Parquet 是用于保存表格数据的列式文件格式。输出只含题号与题面，不含答案。

依赖文件固定的是 PyArrow 版本 18.1.0；推荐使用 Python 3.9–3.13，因为该 PyArrow 版本的官方包元数据只覆盖这些 Python 版本。

## 历史冻结身份

SHA-256 是用于验证文件或数据是否逐字节一致的 256-bit 摘要，通常写成 64 个十六进制字符。下面三个摘要分别锁定内部清单、任务集合和公开清单。

- 完整内部冻结清单的 SHA-256：`5ff786c944a8bb151531082f2b197f9cae45e5e3ebf71848c0e27cb717d873f9`
- 300 个冻结任务哈希按清单顺序组成的集合哈希：`b0beacae2973ff00a30af0b36a699d1e12bfaf29d386aad8b599ed252f2344d6`
- 不含答案的公开清单 SHA-256：`703a6e5a9073f2b434a175eeefbb48a7ec4024027c6cf9e7f29846d3c6d74eec`
- 冻结协议内部编号：`qwen38-q8-q4-v1`。其中 `qwen38` 表示 Qwen 3.8 模型系列，`q8` 与 `q4` 表示最初对照的 8-bit 与 4-bit 权重量化配置，`v1` 表示第一版协议。
