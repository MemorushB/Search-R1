# Load env file to read openai api key
if [ -f ".env" ]; then
  set -a # automatically export all variables
  source .env
  set +a # stop automatically exporting
fi

api_key_to_use=${openai_api_key:-$OPENAI_API_KEY}

corpus_file=data/echr_corpus_sliding_window/echr_corpus_split_512_0.0.jsonl
# corpus_file=data/echr_corpus_split.jsonl
save_dir=data/echr_corpus_sliding_window/512_0.0/bm25 # directory to save the index
retriever_name=bm25 # this is for indexing naming
# retriever_model=intfloat/e5-large-v2
retriever_model=Qdrant/bm25
# retriever_model=text-embedding-3-small
# retriever_model=BAAI/bge-large-en-v1.5


# change faiss_type to HNSW32/64/128 for ANN indexing
# retriever_name option: bm25, openai, bge, e5
# pooling method: cls for BGE, mean for E5
python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method cls \
    --faiss_type Flat \
    --openai_api_key "$api_key_to_use" \
    --openai_model $retriever_model \
    --save_embedding
