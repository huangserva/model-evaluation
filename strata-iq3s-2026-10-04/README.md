# Strata + Qwen3.8-Flash-Next IQ3_S 双轮补测与历史参考榜

本批输出预算 65536 tokens，包含推理及正文。温度 1.0、top_p 0.95、top_k 20、presence_penalty 1.5，推理请求 xhigh，INT8 KV，Strata 在重复推理达到检测阈值后保留全部已生成内容，将原生思考档位从 xhigh 转为 low；未设置思考预算，未强制关闭思考。修复配置单独报告，不替换历史答卷。

独立文字裁判：gpt-5.6-sol，请求 max 档位；每个判分请求不携带其他判分会话，候选答案匿名，工具关闭。每维五项分数 × 0.4，重大违规封顶 3；第二轮每维取三次中位数后相加。

历史模型保持原成绩；批次、渠道、裁判和预算不同，小分差不能视为显著能力差。UltraCode 是工作流，DeepSeek V4.1 Flash 采用修订成绩。

## 第1轮历史参考排行

| 排序 | 模型／配置 | 分数 /100 |
|---:|---|---:|
| 1 | GPT-6 Astra Ultra | 99.6 |
| 2 | GPT-5.6 Sol Ultra / Max | 98.4 |
| 3 | Claude Fable 5 | 97.6 |
| 4 | Hy4 Preview | 96.4 |
| 5 | Kimi-K3 | 95.6 |
| 6 | Claude Fable 5 UltraCode 工作流 | 94.8 |
| 7 | Strata + Qwen3.8-Flash-Next IQ3_S | 94.0 |
| 8 | GLM-5.3 | 93.2 |
| 9 | DeepSeek V4 Pro 0813 | 91.6 |
| 10 | DeepSeek V4.1 Flash | 90.8 |
| 11 | Grok 4.6 | 89.6 |
| 12 | Claude Opus 4.6 | 70.8 |
| 13 | GLM-5.2 | 59.0 |

## 第2轮历史参考排行

| 排序 | 模型／配置 | 分数 /100 |
|---:|---|---:|
| 1 | GPT-6 Astra Ultra | 98.8 |
| 2 | Strata + Qwen3.8-Flash-Next IQ3_S | 96.8 |
| 3 | GPT-5.6 Sol Ultra / Max | 95.2 |
| 4 | GLM-5.3 | 93.6 |
| 5 | DeepSeek V4.1 Flash | 88.2 |
| 6 | Kimi-K3 | 87.4 |
| 7 | Claude Fable 5 UltraCode 工作流 | 85.4 |
| 8 | Claude Opus 4.6 | 85.2 |
| 9 | Grok 4.6 | 84.0 |
| 10 | Hy4 Preview | 83.8 |
| 11 | DeepSeek V4 Pro 0813 | 74.2 |

| 维度 | 三次得分 | 中位数 | 范围 |
|---|---|---:|---|
| D1 | [10, 10, 10] | 10.0 | [10, 10] |
| D2 | [10, 9.2, 10] | 10.0 | [9.2, 10] |
| D3 | [8.8, 9.6, 9.6] | 9.6 | [8.8, 9.6] |
| D4 | [9.6, 9.2, 9.6] | 9.6 | [9.2, 9.6] |
| D5 | [9.6, 9.2, 9.2] | 9.2 | [9.2, 9.6] |
| D6 | [10, 9.6, 10] | 10.0 | [9.6, 10] |
| D7 | [8.4, 8.8, 8.8] | 8.8 | [8.4, 8.8] |
| D8 | [10, 9.6, 10] | 10.0 | [9.6, 10] |
| D9 | [9.6, 9.6, 9.2] | 9.6 | [9.2, 9.6] |
| D10 | [9.6, 10, 10] | 10.0 | [9.6, 10] |

公开的最终答卷及结束原因见 [answers.jsonl](./answers.jsonl)，分数和协议见 [scores.json](./scores.json)，完整性哈希见 [manifest.json](./manifest.json)，修复验证见 [fix-verification.md](./fix-verification.md)。私有评分表、匿名映射、裁判原始备注及隐藏思考不公开。

历史成绩来源：原 2026-08-29 双轮总表、GPT-6 Astra 2026-09-05 补测、GPT-5.6 Sol 第二轮补测及 DeepSeek V4.1 Flash 2026-09-12 修订复核。
