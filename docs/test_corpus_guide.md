# ECHR Corpus Case Retrieval Test - 完整指南

## 概述

`test_echr_corpus.py` 是专门用于测试 Search-R1 模型案例检索性能的自动化评估脚本。该脚本评估模型是否能正确检索到与问题相关的 ECHR 案例，并分析这些案例在检索结果中的排名位置。

## 🎯 核心功能

### 主要测试目标
- **案例ID匹配**: 检查检索器是否返回了正确的案例ID (基于citations中的case_id)
- **排名分析**: 分析正确案例在检索结果中的排名位置
- **性能指标**: 计算Top-K命中率、召回率等关键指标

### 📊 评估指标
1. **Case Retrieval Score**: 每个问题找到的正确案例比例
2. **Perfect Retrieval Rate**: 完美检索(找到所有目标案例)的问题比例  
3. **Case Recall**: 全局案例召回率
4. **Top-K Hit Rate**: 正确案例在Top-1/3/5/10中的出现率
5. **Average/Median Rank**: 正确案例的平均/中位数排名
6. **答案质量评估**: 生成答案与预期答案的相似度对比

### 数据源

- **问题来源**: `data/echr_qa/echr_qa_compact.json`
- **数据库**: `data/echr_corpus_sliding_window/echr_corpus_split_512_0.0.jsonl`
- **真值标准**: QA数据中的 `citations` 字段中的 `case_id`

## 📋 数据流程

```
Input: echr_qa_compact.json
├── question: "How does the Court determine..."
├── citations: [{"case_id": "001-45580", ...}]
└── answer: "The Court considers..."

Processing:
├── 1. 提取问题和目标案例ID
├── 2. 运行Search-R1推理
├── 3. 分析检索结果中的案例ID
└── 4. 计算排名和匹配指标

Output: 
├── JSON: 完整结果数据
├── CSV: 汇总和详细分析表
└── Console: 实时进度和最终统计
```

## 🚀 快速开始

### 1. 一键运行 (推荐)
```bash
./run_corpus_test.sh
```

### 2. Python直接运行
```bash
python test_echr_corpus.py
```

### 3. 修改配置

编辑 `main()` 函数中的参数：

```python
# 模型选择
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3"  # 小模型 (14B参数)
# model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3"  # 大模型 (32B参数)

# 测试样本数量
max_samples = 3  # 开始时使用小数量测试

# 检索参数
topk = 15  # 模型使用的检索数量
evaluation_topk = 100  # 评估时查看的检索结果数量
```

### 4. 自定义数据库配置

```python
evaluator = ECHRCorpusTestEvaluator(
    corpus_path="data/echr_corpus_sliding_window/echr_corpus_split_1024_0.0.jsonl",  # 使用不同的数据库
    evaluation_topk=200  # 增加评估范围
)
```

## 📊 输出结果

### 1. 控制台输出

实时显示每个QA项目的评估进度：

```
=== Evaluating QA Item 1 ===
Question: How does the Court determine whether a surveillance measure...
Target case IDs: ['001-45580']
✅ Generated answer length: 245
📊 Answer similarity: 0.756
📋 Target cases found: 1/1
📈 Case retrieval score: 1.000
🔍 Search queries: 2
⏱️  Duration: 45.3s
📍 Target case rankings:
   - 001-45580: Rank 3
```

### 2. 文件结构

结果保存在 `test_results/` 目录下：

```
test_results/
├── runs/
│   └── corpus_run_20250801_143052/
│       ├── echr_corpus_results_20250801_143052.json  # 完整结果
│       ├── echr_corpus_summary_20250801_143052.csv   # 汇总表
│       ├── echr_corpus_details_20250801_143052.csv   # 详细结果
│       └── intermediate/                              # 中间结果
└── latest/
    ├── echr_corpus_results_latest.json              # 最新结果
    ├── echr_corpus_summary_latest.csv               # 最新汇总
    └── echr_corpus_details_latest.csv               # 最新详细
```

### 3. 最终统计摘要

```
🎯 === Final Evaluation Summary ===
📊 Samples: 3/3
📋 Case Retrieval Score: 0.667
🎯 Perfect Retrieval Rate: 0.333
📈 Case Recall: 0.667
📍 Average Rank: 5.2
🥇 Top-1 Hit Rate: 0.200
🥉 Top-5 Hit Rate: 0.600
📝 Answer Similarity: 0.734
⏱️  Average Duration: 42.8s
```

## 📈 关键评估指标说明

### Case Retrieval Score
- 每个QA项目找到的正确案例数 / 该项目的目标案例总数
- 完美检索率 (Perfect Retrieval Rate): 所有目标案例都被找到的QA项目比例

### Case Recall
- 所有找到的正确案例数 / 所有目标案例总数
- 全局案例召回率

### Top-K Hit Rate
- **Top-1**: 正确案例排在第1位的比例
- **Top-3**: 正确案例排在前3位的比例
- **Top-5**: 正确案例排在前5位的比例
- **Top-10**: 正确案例排在前10位的比例

### 排名分析
- **Average Rank**: 正确案例的平均排名位置
- **Median Rank**: 正确案例的中位数排名位置

## 🔧 高级配置选项

### 模型配置
```python
# 支持不同大小的模型
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3"  # 14B参数
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3"  # 32B参数

# 量化选项
use_quantization = True
quantization_bits = 8  # 或 4
```

### 数据库选择
```python
# 支持不同的corpus数据库
corpus_path = "data/echr_corpus_sliding_window/echr_corpus_split_512_0.0.jsonl"   # 512窗口
corpus_path = "data/echr_corpus_sliding_window/echr_corpus_split_1024_0.0.jsonl"  # 1024窗口
```

### 检索服务器配置
```python
results = evaluator.run_evaluation(
    model_id=model_id,
    max_samples=max_samples,
    topk=topk,
    retrieval_server_url="http://192.168.1.100:8000"  # 使用远程检索服务器
)
```

## 📋 与 test_echr_guide.py 的对比

| 特性 | test_echr_guide.py | test_echr_corpus.py |
|------|-------------------|-------------------|
| **测试目标** | Guide文档段落检索 | 案例(Case)检索 |
| **数据库** | echr_guide.jsonl | echr_corpus_sliding_window/ |
| **匹配标准** | guide_id + paragraph_id | case_id精确匹配 |
| **真值来源** | paragraphs字段 | citations字段 |
| **重点分析** | 段落准确性 | 案例排名位置 |
| **输出重点** | 段落匹配率 | Top-K命中率 |

## 📦 依赖要求

### 必需依赖
- Python 3.8+
- transformers
- torch  
- pandas
- numpy
- requests
- tqdm

### 可选依赖
- sentence-transformers (用于答案相似度计算)

### 运行环境
- GPU推荐 (模型推理)
- 检索服务器运行在 http://127.0.0.1:8000

## 🛠️ 故障排除

### 常见问题

#### 1. 检索服务器连接失败
确保检索服务器正在运行：
```bash
# 检查服务器状态
curl http://127.0.0.1:8000/health

# 启动检索服务器
./retrieval_launch.sh
```

#### 2. GPU内存不足
```python
# 使用较小模型
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3"

# 启用量化
use_quantization = True
quantization_bits = 8
```

#### 3. 数据文件缺失
确保数据文件存在：
```bash
# 检查文件
ls -la data/echr_qa/echr_qa_compact.json
ls -la data/echr_corpus_sliding_window/echr_corpus_split_512_0.0.jsonl
```

### 性能优化建议

1. **首次测试**: 使用 `max_samples=3` 进行快速验证
2. **完整评估**: 增加到 `max_samples=20+` 进行全面测试
3. **预加载模型**: 设置 `preload_model=True` 避免重复加载
4. **批量保存**: 每5个样本保存中间结果，避免数据丢失

## 🎯 项目文件组织

```
/srv/chenru/Search-R1/
├── test_echr_corpus.py              # 主测试脚本 (711行)
├── run_corpus_test.sh              # 快速运行脚本
├── README_corpus_test.md           # 项目说明
├── docs/
│   └── test_corpus_guide.md        # 详细使用指南
└── test_results/                   # 结果目录(自动创建)
    ├── runs/                       # 历史运行记录
    ├── latest/                     # 最新结果
    └── logs/                       # 日志文件
```

## ✅ 功能特性总结

### 测试评估能力
- ✅ **案例ID匹配**: 基于 `citations` 字段中的 `case_id` 进行精确匹配
- ✅ **排名分析**: 分析正确案例在检索结果中的排名位置(1-100)
- ✅ **多维度指标**: Top-K命中率、召回率、完美检索率等
- ✅ **实时监控**: 提供详细的进度报告和中间结果保存

### 数据处理流程
- ✅ 数据加载和解析
- ✅ 模型推理集成
- ✅ 检索结果分析
- ✅ 排名计算算法
- ✅ 多格式结果输出
- ✅ 中间结果保存
- ✅ 错误处理和恢复

### 用户体验
- ✅ 实时进度显示
- ✅ 详细的帮助文档
- ✅ 一键运行脚本
- ✅ 故障排除指南

## 🔗 相关文件

- 详细项目说明: [README_corpus_test.md](../README_corpus_test.md)
- 参考实现: [test_echr_guide.py](../test_echr_guide.py)
- 推理引擎: [inference_engine.py](../inference_engine.py)
- 项目总结: [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)
