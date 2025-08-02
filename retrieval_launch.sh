#!/bin/bash

# Model selection: Choose 1 for bm25, 2 for e5, 3 for openai, 4 for bge with reranker, 5 for sbert
model_choice=${1:-1}  # Default to 1 (bm25) if no argument provided

file_path=data
corpus_file=$file_path/echr_corpus_sliding_window/echr_corpus_split_512_0.0.jsonl

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
        index_file="$file_path/echr_corpus_sliding_window/512_0.0/bm25/bm25"
        ;;
    2)  # E5
        echo "Using E5 model"
        retriever_name="e5"
        retriever_path="intfloat/e5-large-v2"
        index_file="$file_path/echr_corpus_sliding_window/512_0.0/e5/e5_Flat.index"
        ;;
    3)  # OpenAI
        echo "Using OpenAI model"
        retriever_name="openai"
        retriever_path="text-embedding-3-small"
        index_file="$file_path/echr_corpus_sliding_window/512_0.0/openai/openai_Flat.index"
        ;;
    4)  # BGE with reranker
        echo "Using BGE model with BGE reranker"
        retriever_name="bge"
        retriever_path="BAAI/bge-large-en-v1.5"
        reranker_path="BAAI/bge-reranker-v2-m3"
        use_reranker=true
        index_file="$file_path/echr_corpus_sliding_window/512_0.0/bge/bge_Flat.index"
        ;;
    5)  # SBERT
        echo "Using Sentence-BERT model"
        retriever_name="sbert"
        retriever_path="sentence-transformers/all-MiniLM-L6-v2"
        index_file="$file_path/echr_corpus_sliding_window/512_0.0/sbert/sbert_Flat.index"
        ;;
    *)
        echo "Invalid model choice. Please select 1, 2, 3, 4, or 5."
        echo "1: BM25"
        echo "2: E5"
        echo "3: OpenAI"
        echo "4: BGE with BGE reranker"
        echo "5: Sentence-BERT"
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

RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-250}  

# Build command based on selected model
cmd="python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk $RETRIEVAL_TOPK \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --openai_api_key $openai_api_key"

# Add reranker option only if using BGE with reranker
if [ "$use_reranker" = true ]; then
    cmd="$cmd --reranker_model $reranker_path"
fi

# Add FAISS GPU support
cmd="$cmd --faiss_gpu"

echo "Using $retriever_name retriever with topk=$RETRIEVAL_TOPK"
echo "Running command:"
echo $cmd
eval $cmd