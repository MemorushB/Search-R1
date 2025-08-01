# 🚀 Search-R1 模型预加载优化与测试指南

## 📖 概述

本文档包含Search-R1项目的模型预加载优化说明和完整的测试使用指南。最新的优化确保了在运行评估时，模型只需要加载一次，而不是每个样本都重新加载，大大提高了评估效率。

---

## 🎯 模型预加载优化

### 修改内容

#### 1. inference_engine.py 重构
- **新增 SearchR1Model 类**: 支持模型的持久化加载和复用
- **全局模型管理**: `get_or_create_model()` 函数提供模型单例管理
- **保持向后兼容**: 原有的 `search_r1_inference()` 函数仍可使用，内部使用新的预加载机制

#### 2. test_echr_guide.py 增强
- **新增预加载选项**: ECHRTestEvaluator 构造函数支持模型预加载参数
  - `preload_model`: 是否启用模型预加载
  - `model_id`: 指定模型ID
  - `use_quantization`: 是否使用量化
  - `quantization_bits`: 量化位数 (4或8)
- **智能推理调用**: 自动选择预加载模型或标准函数进行推理

#### 3. run_evaluation.py 命令行增强
- **新增预加载选项**:
  - `--no-preload`: 禁用模型预加载
  - `--no-quantization`: 禁用量化
  - `--quantization-bits`: 设置量化位数
- **参数传递**: 将预加载配置传递给ECHRTestEvaluator

### 主要优势

#### 性能提升
- **减少重复加载**: 模型只在首次使用时加载一次，后续复用
- **内存优化**: 避免每个样本都重新分配GPU内存
- **显著提速**: 对于多样本评估，可大幅减少总时间

#### 用户体验
- **向后兼容**: 现有代码无需修改即可获得性能提升
- **灵活配置**: 可根据需要启用/禁用预加载
- **详细反馈**: 提供模型加载状态和配置信息

### 技术实现

#### 全局模型管理
- 使用全局变量 `_global_model` 存储模型实例
- `get_or_create_model()` 检查模型ID是否变化，只在必要时重新加载
- 自动GPU内存清理，避免内存泄漏

#### 类设计
- `SearchR1Model` 类封装模型加载和推理逻辑
- 支持量化配置 (4-bit/8-bit)
- 设备自动检测和优化

#### 错误处理
- 模型加载失败时自动回退到标准模式
- 提供详细的错误信息和状态反馈

---

## 📂 测试文件概览

### 核心测试文件
- `test_basic.py` - 基础功能测试
- `test_retriever_info.py` - Retriever信息检查
- `test_echr_guide.py` - 完整ECHR QA评估框架（已优化支持预加载）
- `run_evaluation.py` - 命令行评估工具（已增强，支持预加载配置）
- `test_basic_preload.py` - 基础预加载功能测试
- `test_preload.py` - 完整预加载功能测试

### 支持文件  
- `inference_engine.py` - 推理引擎（已重构，支持模型预加载）
- `retrieval_launch.sh` - Retrieval服务器启动脚本

---

## 🎯 使用方法

### 1. 准备工作

**启动Retrieval服务器**：
```bash
# 方法1：使用脚本启动（推荐）
# 选择不同的模型配置：
# 1. BM25: bash retrieval_launch.sh 1
# 2. E5: bash retrieval_launch.sh 2
# 3. OpenAI: bash retrieval_launch.sh 3
# 4. BGE + BGE reranker: bash retrieval_launch.sh 4
bash retrieval_launch.sh 4  # 默认使用BGE + reranker

# 方法2：手动启动（在searchr1环境中）
conda activate searchr1
python search_r1/search/retrieval_server.py --index_path data/echr_guide_index/bge/bge_Flat.index --corpus_path data/echr_guide.jsonl --topk 15 --retriever_name bge --retriever_model BAAI/bge-large-en-v1.5 --reranker_model BAAI/bge-reranker-v2-m3 --faiss_gpu
```

### 2. 快速测试

**基础功能检查**：
```bash
python test_basic.py
```

**Retriever信息检查**：
```bash
python test_retriever_info.py
```

**模型预加载功能测试**：
```bash
python test_basic_preload.py
```

### 3. 模型预加载使用方式

#### 方式A：直接使用新类（推荐）
```python
from inference_engine import SearchR1Model

# 创建预加载模型
model = SearchR1Model(
    model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-grpo-v0.3",
    use_quantization=True,
    quantization_bits=8
)

# 多次推理
for question in questions:
    answer, log, meta = model.inference(question=question)
```

#### 方式B：使用增强的评估器
```python
from test_echr_guide import ECHRTestEvaluator

# 启用预加载
evaluator = ECHRTestEvaluator(
    preload_model=True,
    model_id="your-model-id",
    use_quantization=True
)
```

#### 方式C：使用命令行工具（最便捷）
```bash
# 启用预加载 (默认)
python run_evaluation.py --samples 10

# 禁用预加载
python run_evaluation.py --samples 10 --no-preload

# 自定义量化
python run_evaluation.py --samples 10 --quantization-bits 4
```

### 4. ECHR QA评估

#### 快速测试（使用预加载优化）

**快速测试（2个样本）**：
```bash
python run_evaluation.py --quick
```

**中等测试（10个样本）**：
```bash
python run_evaluation.py --medium
```

**完整测试（所有样本）**：
```bash
python run_evaluation.py --full
```

**使用14B模型**：
```bash
python run_evaluation.py --model-14b --quick
```

**自定义参数（含预加载配置）**：
```bash
python run_evaluation.py --samples 5 --topk 10 --model-14b --quantization-bits 4
```

#### 传统方法（兼容性）

**直接运行**：
```bash
python test_echr_guide.py
```

### 5. 高级用法

**指定自定义数据路径**：
```bash
python run_evaluation.py --qa-data /path/to/qa.json --guide-data /path/to/guide.jsonl --results-dir /path/to/results
```

**使用不同的retrieval服务器**：
```bash
python run_evaluation.py --server-url http://192.168.1.100:8000
```

**完全自定义配置**：
```bash
python run_evaluation.py \
  --samples 20 \
  --topk 15 \
  --model "custom-model-id" \
  --quantization-bits 8 \
  --results-dir custom_results \
  --server-url http://localhost:8000
```

---

## 🔧 参数说明

### run_evaluation.py 参数（已增强）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 32B模型 | 指定模型ID |
| `--samples` | 3 | 测试样本数量 |
| `--topk` | 15 | 检索TopK值 |
| `--qa-data` | data/echr_qa/echr_qa_compact.json | QA数据路径 |
| `--guide-data` | data/echr_guide.jsonl | Guide数据路径 |
| `--results-dir` | test_results | 结果保存目录 |
| `--server-url` | http://127.0.0.1:8000 | Retrieval服务器URL |
| `--quick` | - | 快速测试（2个样本） |
| `--medium` | - | 中等测试（10个样本） |
| `--full` | - | 完整测试（所有样本） |
| `--model-14b` | - | 使用14B模型 |
| `--no-preload` | - | **新增**: 禁用模型预加载 |
| `--no-quantization` | - | **新增**: 禁用量化 |
| `--quantization-bits` | 8 | **新增**: 设置量化位数 (4或8) |

---

## 🎛️ 常用测试场景

### 开发调试（快速，使用预加载）
```bash
python run_evaluation.py --quick --model-14b
```

### 性能评估（中等规模，优化版）
```bash
python run_evaluation.py --medium
```

### 完整验证（全量测试）
```bash
python run_evaluation.py --full
```

### 自定义测试（指定配置）
```bash
python run_evaluation.py --samples 8 --topk 20 --results-dir custom_results --quantization-bits 4
```

### 性能对比测试
```bash
# 不使用预加载
python run_evaluation.py --quick --no-preload

# 使用预加载（默认）
python run_evaluation.py --quick
```

---

## 📊 结果文件

评估完成后，会生成以下文件：

### 在 `test_results/` 目录中：
- `echr_qa_results_YYYYMMDD_HHMMSS.json` - 详细结果（包含所有答案和元数据）
- `echr_qa_summary_YYYYMMDD_HHMMSS.csv` - 汇总统计（包含retriever信息）
- `echr_qa_details_YYYYMMDD_HHMMSS.csv` - 详细数据表
- `intermediate_results_N_YYYYMMDD_HHMMSS.json` - 中间结果（每2个样本保存一次）

### 在 `test_results/inference_logs/` 目录中：
- `inference_log_YYYYMMDD_HHMMSS.txt` - 推理过程日志

---

## 📈 结果解读

### 关键指标：
- **Success Rate**: 成功完成推理的样本比例
- **Average Similarity**: 与期望答案的平均相似度
- **Average Search Count**: 平均搜索次数
- **Average Retrieval Score**: 平均检索得分
- **Retriever Info**: 使用的检索系统信息

### Retriever信息包含：
- `retriever_type`: 检索器类型（如"bge"）
- `model_name`: 模型名称（如"BAAI/bge-large-en-v1.5"）
- `index_type`: 索引类型（如"FAISS_Flat"）
- `reranker_model`: 重排模型（如"BAAI/bge-reranker-v2-m3"）

---

## 🚀 性能优化效果

### 预期性能提升：
- **首次推理**: 正常加载时间
- **后续推理**: 显著加速 (节省模型加载时间)
- **内存使用**: 稳定，无泄漏
- **总体效率**: 多样本评估可提速50-80%

### 测试验证：

**提供的测试脚本**：
1. `test_basic_preload.py`: 基础功能测试
2. `test_preload.py`: 完整功能测试，包括评估器集成

**性能基准测试**：
```bash
# 对比测试：不使用预加载 vs 使用预加载
time python run_evaluation.py --quick --no-preload
time python run_evaluation.py --quick
```

---

## ✅ 兼容性保证

- ✅ 与现有代码完全兼容
- ✅ 支持所有SearchR1模型变体
- ✅ 支持量化和非量化模式
- ✅ 支持多种硬件配置
- ✅ 向后兼容原有测试脚本

---

## 🚨 注意事项

1. **确保Retrieval服务器正在运行**，否则测试会失败
2. **32B模型需要大量GPU内存**，建议使用量化或14B模型
3. **第一次运行会下载模型**，需要时间和网络
4. **结果文件会自动按时间戳命名**，不会覆盖之前的结果
5. **模型预加载默认启用**，如需禁用使用`--no-preload`参数
6. **量化默认启用8-bit**，可通过`--quantization-bits`调整或`--no-quantization`禁用

---

## 📞 故障排除

### 常见问题：

**Q: 预加载失败怎么办？**
A: 系统会自动回退到标准模式，无需担心

**Q: 内存不足怎么办？**
A: 使用`--model-14b`或`--quantization-bits 4`减少内存使用

**Q: 如何验证预加载是否生效？**
A: 运行`python test_basic_preload.py`进行验证

**Q: 想使用原来的方式怎么办？**
A: 使用`--no-preload`参数或直接运行`python test_echr_guide.py`

这个优化确保了在运行 `run_evaluation.py` 时，模型只需要加载一次，而不是每个样本都重新加载，大大提高了评估效率！🎉
