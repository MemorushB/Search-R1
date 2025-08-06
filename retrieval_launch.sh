#!/bin/bash

# Model selection: Choose 1 for bm25, 2 for e5, 3 for openai, 4 for bge with reranker, 5 for sbert, 6 for qwen, 7 for qwen with reranker
model_choice=${1:-1}  # Default to 1 (bm25) if no argument provided

index_path=data/echr_corpus_sliding_window/512_0.1
corpus_file=data/echr_corpus_sliding_window/echr_corpus_split_512_0.1.jsonl

# Set default values
retriever_name="sbert"
retriever_path="sentence-transformers/all-MiniLM-L6-v2"
reranker_path=""
use_reranker=false

# Parse command line arguments
case $model_choice in
    1)  # BM25
        echo "Using BM25 model"
        retriever_name="bm25"
        retriever_path="Qdrant/bm25"
        index_file="$index_path/bm25/bm25"
        ;;
    2)  # E5
        echo "Using E5 model"
        retriever_name="e5"
        retriever_path="intfloat/e5-large-v2"
        index_file="$index_path/e5/e5_Flat.index"
        ;;
    3)  # OpenAI
        echo "Using OpenAI model"
        retriever_name="openai"
        retriever_path="text-embedding-3-small"
        index_file="$index_path/openai/openai_Flat.index"
        ;;
    4)  # BGE with reranker
        echo "Using BGE model with BGE reranker"
        retriever_name="bge"
        retriever_path="BAAI/bge-m3"
        reranker_path="BAAI/bge-reranker-v2-m3"
        use_reranker=true
        index_file="$index_path/bge/bge_Flat.index"
        ;;
    5)  # SBERT
        echo "Using Sentence-BERT model"
        retriever_name="sbert"
        retriever_path="sentence-transformers/all-MiniLM-L6-v2"
        index_file="$index_path/sbert/sbert_Flat.index"
        ;;
    6)  # Qwen
        echo "Using Qwen model"
        retriever_name="qwen"
        retriever_path="Qwen/Qwen3-Embedding-0.6B"
        index_file="$index_path/qwen/qwen_Flat.index"
        ;;
    7)  # Qwen with reranker
        echo "Using Qwen model with Qwen reranker"
        retriever_name="qwen"
        retriever_path="Qwen/Qwen3-Embedding-0.6B"
        reranker_path="Qwen/Qwen3-Reranker-0.6B"
        use_reranker=true
        index_file="$index_path/qwen/qwen_Flat.index"
        ;;
    *)
        echo "Invalid model choice. Please select 1, 2, 3, 4, 5, 6, or 7."
        echo "1: BM25"
        echo "2: E5"
        echo "3: OpenAI"
        echo "4: BGE with BGE reranker"
        echo "5: Sentence-BERT"
        echo "6: Qwen"
        echo "7: Qwen with Qwen reranker"
        exit 1
        ;;
esac


# Load .env file if it exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Get OpenAI API key from environment
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not found"
    echo "📝 Please either:"
    echo "   1. Create .env file: echo 'OPENAI_API_KEY=your_key' > .env"
    echo "   2. Set environment: export OPENAI_API_KEY=your_key"
    exit 1
fi

openai_api_key="$OPENAI_API_KEY"
echo "✅ Using OpenAI API key (${openai_api_key:0:10}...)"

RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-800}  

# Build command based on selected model
cmd="python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk $RETRIEVAL_TOPK \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --openai_api_key $openai_api_key"

# Add reranker option only if using BGE with reranker or Qwen with reranker
if [ "$use_reranker" = true ]; then
    cmd="$cmd --reranker_model $reranker_path"
fi

# Add FAISS GPU support
cmd="$cmd --faiss_gpu"

echo "Using $retriever_name retriever with topk=$RETRIEVAL_TOPK"
if [ "$use_reranker" = true ]; then
    echo "With reranker: $reranker_path"
fi
echo "Running command:"
echo $cmd
eval $cmd