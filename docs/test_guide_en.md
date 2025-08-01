# 🚀 Search-R1 Model Preloading Optimization & Testing Guide

## 📖 Overview

This document contains the model preloading optimization details and comprehensive testing guide for the Search-R1 project. The latest optimization ensures that when running evaluations, the model only needs to be loaded once instead of reloading for each sample, significantly improving evaluation efficiency.

---

## 🎯 Model Preloading Optimization

### Changes Made

#### 1. inference_engine.py Refactoring
- **Added SearchR1Model Class**: Supports persistent model loading and reuse
- **Global Model Management**: `get_or_create_model()` function provides singleton model management
- **Backward Compatibility**: Original `search_r1_inference()` function remains usable, internally using the new preloading mechanism

#### 2. test_echr_guide.py Enhancement
- **Added Preloading Options**: ECHRTestEvaluator constructor supports model preloading parameters
  - `preload_model`: Whether to enable model preloading
  - `model_id`: Specify model ID
  - `use_quantization`: Whether to use quantization
  - `quantization_bits`: Quantization bits (4 or 8)
- **Smart Inference Calling**: Automatically chooses preloaded model or standard function for inference

#### 3. run_evaluation.py Command-line Enhancement
- **Added Preloading Options**:
  - `--no-preload`: Disable model preloading
  - `--no-quantization`: Disable quantization
  - `--quantization-bits`: Set quantization bits
- **Parameter Passing**: Pass preloading configuration to ECHRTestEvaluator

### Key Advantages

#### Performance Improvement
- **Reduced Repeated Loading**: Model loads only once on first use, then reused
- **Memory Optimization**: Avoids GPU memory reallocation for each sample
- **Significant Speedup**: For multi-sample evaluations, can dramatically reduce total time

#### User Experience
- **Backward Compatible**: Existing code gains performance improvements without modification
- **Flexible Configuration**: Can enable/disable preloading as needed
- **Detailed Feedback**: Provides model loading status and configuration information

### Technical Implementation

#### Global Model Management
- Uses global variable `_global_model` to store model instance
- `get_or_create_model()` checks if model ID changes, only reloads when necessary
- Automatic GPU memory cleanup to prevent memory leaks

#### Class Design
- `SearchR1Model` class encapsulates model loading and inference logic
- Supports quantization configuration (4-bit/8-bit)
- Automatic device detection and optimization

#### Error Handling
- Automatically falls back to standard mode when model loading fails
- Provides detailed error information and status feedback

---

## 📂 Test Files Overview

### Core Test Files
- `test_basic.py` - Basic functionality testing
- `test_retriever_info.py` - Retriever information checking
- `test_echr_guide.py` - Complete ECHR QA evaluation framework (optimized with preloading support)
- `run_evaluation.py` - Command-line evaluation tool (enhanced with preloading configuration support)
- `test_basic_preload.py` - Basic preloading functionality testing
- `test_preload.py` - Complete preloading functionality testing

### Support Files  
- `inference_engine.py` - Inference engine (refactored with model preloading support)
- `retrieval_launch.sh` - Retrieval server launch script

---

## 🎯 Usage

### 1. Preparation

**Start Retrieval Server**:
```bash
# Method 1: Use script to start (Recommended)
# Choose different model configurations:
# 1. BM25: bash retrieval_launch.sh 1
# 2. E5: bash retrieval_launch.sh 2
# 3. OpenAI: bash retrieval_launch.sh 3
# 4. BGE + BGE reranker: bash retrieval_launch.sh 4
bash retrieval_launch.sh 4  # Default to BGE + reranker

# Method 2: Manual start (in searchr1 environment)
conda activate searchr1
python search_r1/search/retrieval_server.py --index_path data/echr_guide_index/bge/bge_Flat.index --corpus_path data/echr_guide.jsonl --topk 15 --retriever_name bge --retriever_model BAAI/bge-large-en-v1.5 --reranker_model BAAI/bge-reranker-v2-m3 --faiss_gpu
```

### 2. Quick Testing

**Basic functionality check**:
```bash
python test_basic.py
```

**Retriever information check**:
```bash
python test_retriever_info.py
```

**Model preloading functionality test**:
```bash
python test_basic_preload.py
```

### 3. Model Preloading Usage

#### Method A: Direct use of new class (Recommended)
```python
from inference_engine import SearchR1Model

# Create preloaded model
model = SearchR1Model(
    model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-grpo-v0.3",
    use_quantization=True,
    quantization_bits=8
)

# Multiple inferences
for question in questions:
    answer, log, meta = model.inference(question=question)
```

#### Method B: Use enhanced evaluator
```python
from test_echr_guide import ECHRTestEvaluator

# Enable preloading
evaluator = ECHRTestEvaluator(
    preload_model=True,
    model_id="your-model-id",
    use_quantization=True
)
```

#### Method C: Use command-line tool (Most convenient)
```bash
# Enable preloading (default)
python run_evaluation.py --samples 10

# Disable preloading
python run_evaluation.py --samples 10 --no-preload

# Custom quantization
python run_evaluation.py --samples 10 --quantization-bits 4
```

### 4. ECHR QA Evaluation

#### Quick Testing (Using preloading optimization)

**Quick test (2 samples)**:
```bash
python run_evaluation.py --quick
```

**Medium test (10 samples)**:
```bash
python run_evaluation.py --medium
```

**Full test (all samples)**:
```bash
python run_evaluation.py --full
```

**Use 14B model**:
```bash
python run_evaluation.py --model-14b --quick
```

**Custom parameters (with preloading configuration)**:
```bash
python run_evaluation.py --samples 5 --topk 10 --model-14b --quantization-bits 4
```

#### Traditional Method (Compatibility)

**Direct run**:
```bash
python test_echr_guide.py
```

### 5. Advanced Usage

**Specify custom data paths**:
```bash
python run_evaluation.py --qa-data /path/to/qa.json --guide-data /path/to/guide.jsonl --results-dir /path/to/results
```

**Use different retrieval server**:
```bash
python run_evaluation.py --server-url http://192.168.1.100:8000
```

**Fully custom configuration**:
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

## 🔧 Parameter Description

### run_evaluation.py Parameters (Enhanced):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | 32B model | Specify model ID |
| `--samples` | 3 | Number of test samples |
| `--topk` | 15 | Retrieval TopK value |
| `--qa-data` | data/echr_qa/echr_qa_compact.json | QA data path |
| `--guide-data` | data/echr_guide.jsonl | Guide data path |
| `--results-dir` | test_results | Results save directory |
| `--server-url` | http://127.0.0.1:8000 | Retrieval server URL |
| `--quick` | - | Quick test (2 samples) |
| `--medium` | - | Medium test (10 samples) |
| `--full` | - | Full test (all samples) |
| `--model-14b` | - | Use 14B model |
| `--no-preload` | - | **New**: Disable model preloading |
| `--no-quantization` | - | **New**: Disable quantization |
| `--quantization-bits` | 8 | **New**: Set quantization bits (4 or 8) |

---

## 🎛️ Common Testing Scenarios

### Development Debug (Fast, using preloading)
```bash
python run_evaluation.py --quick --model-14b
```

### Performance Evaluation (Medium scale, optimized)
```bash
python run_evaluation.py --medium
```

### Full Validation (Complete test)
```bash
python run_evaluation.py --full
```

### Custom Test (Specify configuration)
```bash
python run_evaluation.py --samples 8 --topk 20 --results-dir custom_results --quantization-bits 4
```

### Performance Comparison Test
```bash
# Without preloading
python run_evaluation.py --quick --no-preload

# With preloading (default)
python run_evaluation.py --quick
```

---

## 📊 Result Files

After evaluation completion, the following files will be generated in the new organized structure:

### In `test_results/` directory:
```
test_results/
├── runs/                           # Historical run records
│   ├── run_20250731_143022/       # Timestamped directory for each run
│   │   ├── results.json           # Complete results with metadata
│   │   ├── summary.csv            # Summary statistics
│   │   ├── details.csv            # Detailed per-sample results
│   │   └── intermediate/          # Intermediate results
│   │       ├── step_002.json      # Step 2 results
│   │       ├── step_004.json      # Step 4 results
│   │       └── ...
│   └── run_20250731_150315/       # Another run
├── logs/                          # Inference logs (unified storage)
└── latest/                        # Quick access to latest results
    ├── results.json
    ├── summary.csv
    └── details.csv
```

### Key improvements in file organization:
- **Clear Directory Organization**: Each run has its own timestamped directory
- **Simple File Naming**: `results.json` instead of `echr_qa_results_20250731_143022.json`
- **Ordered Intermediate Results**: `step_002.json`, `step_004.json` with zero-padded numbers
- **Quick Access**: `latest/` directory always contains copies of the most recent results

---

## 📈 Result Interpretation

### Key Metrics:
- **Success Rate**: Proportion of samples that completed inference successfully
- **Average Similarity**: Average similarity with expected answers
- **Average Search Count**: Average number of searches
- **Average Retrieval Score**: Average retrieval score
- **Retriever Info**: Information about the retrieval system used

### Retriever Information includes:
- `retriever_type`: Retriever type (e.g., "bge")
- `model_name`: Model name (e.g., "BAAI/bge-large-en-v1.5")
- `index_type`: Index type (e.g., "FAISS_Flat")
- `reranker_model`: Reranker model (e.g., "BAAI/bge-reranker-v2-m3")

---

## 🚀 Performance Optimization Effects

### Expected Performance Improvements:
- **First Inference**: Normal loading time
- **Subsequent Inferences**: Significant speedup (saves model loading time)
- **Memory Usage**: Stable, no leaks
- **Overall Efficiency**: Multi-sample evaluation can improve speed by 50-80%

### Testing Verification:

**Provided test scripts**:
1. `test_basic_preload.py`: Basic functionality testing
2. `test_preload.py`: Complete functionality testing, including evaluator integration

**Performance benchmark testing**:
```bash
# Comparison test: without preloading vs with preloading
time python run_evaluation.py --quick --no-preload
time python run_evaluation.py --quick
```

---

## ✅ Compatibility Guarantee

- ✅ Fully compatible with existing code
- ✅ Supports all SearchR1 model variants
- ✅ Supports quantized and non-quantized modes
- ✅ Supports multiple hardware configurations
- ✅ Backward compatible with original test scripts

---

## 🚨 Important Notes

1. **Ensure Retrieval server is running**, otherwise tests will fail
2. **32B model requires significant GPU memory**, recommend using quantization or 14B model
3. **First run will download model**, requires time and network
4. **Result files are automatically named with timestamps**, won't overwrite previous results
5. **Model preloading is enabled by default**, use `--no-preload` parameter to disable
6. **Quantization defaults to 8-bit**, can be adjusted via `--quantization-bits` or disabled with `--no-quantization`

---

## 📞 Troubleshooting

### Common Issues:

**Q: What if preloading fails?**
A: The system will automatically fall back to standard mode, no need to worry

**Q: What if there's insufficient memory?**
A: Use `--model-14b` or `--quantization-bits 4` to reduce memory usage

**Q: How to verify if preloading is working?**
A: Run `python test_basic_preload.py` for verification

**Q: How to use the original method?**
A: Use `--no-preload` parameter or directly run `python test_echr_guide.py`

This optimization ensures that when running `run_evaluation.py`, the model only needs to be loaded once instead of reloading for each sample, significantly improving evaluation efficiency! 🎉
