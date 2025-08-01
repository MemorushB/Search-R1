# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains Search-R1, a reinforcement learning framework for training reasoning-and-searching interleaved LLMs. It enables language models to learn to reason and make tool calls (e.g., to search engines) in a coordinated manner. Built upon veRL, Search-R1 extends the ideas of DeepSeek-R1(-Zero) by incorporating interleaved search engine access and provides a fully open-source RL training pipeline.

## Architecture

The codebase is structured into several key components:

1. **verl/** - Main reinforcement learning framework based on veRL
   - Core training infrastructure with PPO/GRPO implementations
   - Distributed training with Ray
   - Model workers for actor, critic, and reference policies
   - Support for FSDP and Megatron training strategies

2. **search_r1/** - Search engine integration components
   - Retrieval server with multiple search engine backends (BM25, dense retrievers, OpenAI embeddings)
   - API endpoints for retrieval services
   - Indexing and corpus management

3. **Training Pipeline**
   - Data processing scripts for preparing QA datasets
   - Configuration-driven training with Hydra
   - Reward management for rule-based outcome rewards
   - Checkpointing and experiment tracking with Weights & Biases

## Common Commands

### Installation
```bash
# Create conda environment
conda create -n searchr1 python=3.9
conda activate searchr1

# Install dependencies
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3
pip install -e .
pip3 install flash-attn --no-build-isolation
pip install wandb

# Optional retriever environment
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

### Running Retrieval Server
```bash
# Launch retrieval server (separate terminal)
conda activate retriever
bash retrieval_launch.sh
```

### Training
```bash
# Run PPO training
conda activate searchr1
bash train_ppo.sh

# Run GRPO training
bash train_grpo.sh
```

### Inference
```bash
# Run inference with trained model
conda activate searchr1
python infer.py
```

### Testing
```bash
# Run ECHR QA evaluation
python test_echr_guide.py
```

## Key Configuration Files

- `verl/trainer/config/ppo_trainer.yaml` - Main PPO training configuration
- `verl/trainer/config/ppo_megatron_trainer.yaml` - Megatron-specific PPO configuration
- `train_ppo.sh` and `train_grpo.sh` - Training launch scripts with hyperparameters

## Data Flow

1. **Training Data**: QA datasets are processed into structured formats with prompts, ground truth answers, and reward configurations
2. **Model Training**: LLMs are trained using PPO/GRPO with rule-based rewards computed from ground truth answers
3. **Retrieval Integration**: Models can call search APIs during generation to retrieve relevant information
4. **Evaluation**: Models are evaluated on QA tasks with similarity-based metrics and retrieval effectiveness measures

## Key Components

- **RayPPOTrainer**: Main training orchestrator using Ray for distributed training
- **ActorRolloutRefWorker**: Handles model inference, rollout generation, and reference policy computation
- **CriticWorker**: Computes value estimates for policy gradient updates
- **RewardManager**: Computes rewards based on ground truth answers using exact match or other metrics
- **Retrieval Server**: FastAPI service providing search capabilities to models during inference