import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets
import openai  
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Import sentence-transformers for SBERT support
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not installed. SBERT models will not be available.")

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results
        
class OpenAIEncoder:
    """OpenAI Embedding Encoder"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str = None):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.max_tokens = 8191
        
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]
            
        # truncate texts to fit within the token limit
        truncated_texts = []
        for text in query_list:
            if len(text) > self.max_tokens * 4:
                text = text[:self.max_tokens * 4]
            truncated_texts.append(text)

        # call OpenAI API
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                response = self.client.embeddings.create(
                    input=truncated_texts,
                    model=self.model
                )
                
                embeddings = [data.embedding for data in response.data]
                embeddings = np.array(embeddings, dtype=np.float32)
                return embeddings
                
            except openai.RateLimitError:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    print(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                print(f"Error in OpenAI API call: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                time.sleep(1)
                

class BGEEncoder:
    """BGE Embedding Encoder"""
    
    def __init__(self, model_name, model_path, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.model.eval()
        self.model.cuda()
        if use_fp16:
            self.model = self.model.half()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        # Add BGE instruction for queries
        if is_query and "bge" in self.model_name.lower():
            query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to('cuda')

        output = self.model(**inputs, return_dict=True)
        # Use CLS token for BGE
        query_emb = output.last_hidden_state[:, 0]
        # Normalize embeddings
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class SBERTEncoder:
    """Sentence-BERT Embedding Encoder"""
    
    def __init__(self, model_name, model_path, max_length, use_fp16):
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers is not installed. Please install it with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        
        # Load SentenceTransformer model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_path, device=device)
        
        # Set max sequence length
        if hasattr(self.model, 'max_seq_length'):
            self.model.max_seq_length = max_length
        
        # Enable fp16 if specified
        if use_fp16 and device == 'cuda':
            self.model = self.model.half()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        # Some SBERT models might benefit from query/passage prefixes
        # This can be customized based on specific SBERT model requirements
        if "msmarco" in self.model_name.lower() and is_query:
            # For MS MARCO models, you might want to add query prefix
            pass  # No special handling needed for most SBERT models
        
        # Encode using SentenceTransformer
        embeddings = self.model.encode(
            query_list,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize embeddings for better retrieval
            batch_size=32,  # Process in batches for efficiency
            show_progress_bar=False
        )
        
        # Ensure float32 format
        embeddings = embeddings.astype(np.float32, order="C")
        
        return embeddings

class BGEReranker:
    """BGE Reranker for improving retrieval results"""
    
    def __init__(self, model_path: str, use_fp16: bool = False, max_length: int = 512):
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        
        # Load BGE reranker model
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.model.eval()
        self.model.cuda()
        if use_fp16:
            self.model = self.model.half()
    
    @torch.no_grad()
    def rerank(self, query: str, passages: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Rerank passages based on query-passage relevance
        
        Args:
            query: Query string
            passages: List of passage dictionaries with 'contents' field
            top_k: Number of top passages to return after reranking
        
        Returns:
            Reranked list of passages
        """
        if not passages:
            return passages
            
        if top_k is None:
            top_k = len(passages)
        
        # Prepare query-passage pairs
        pairs = []
        for passage in passages:
            content = passage.get('contents', '')
            pairs.append([query, content])
        
        # Batch process for efficiency
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # Tokenize pairs
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length
            ).to('cuda')
            
            # Get relevance scores
            outputs = self.model(**inputs)
            scores = outputs.last_hidden_state[:, 0]  # Use CLS token
            
            # For reranking, we typically use a linear layer or similarity score
            # Here we use the norm of CLS token as relevance score
            scores = torch.norm(scores, dim=-1)
            scores = scores.cpu().numpy()
            all_scores.extend(scores)
        
        # Sort by scores (descending)
        scored_passages = list(zip(passages, all_scores))
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k reranked passages
        reranked_passages = [passage for passage, _ in scored_passages[:top_k]]
        
        return reranked_passages


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        
        # Choose encoder based on retrieval method
        if self.retrieval_method == "openai" or "openai" in self.retrieval_method:
            self.encoder = OpenAIEncoder(
                api_key=config.openai_api_key,
                model=config.retrieval_model_path,
                base_url=getattr(config, 'openai_base_url', None)
            )
        elif self.retrieval_method == "bge" or "bge" in self.retrieval_method:
            self.encoder = BGEEncoder(
                model_name=self.retrieval_method,
                model_path=config.retrieval_model_path,
                max_length=config.retrieval_query_max_length,
                use_fp16=config.retrieval_use_fp16
            )
        elif self.retrieval_method == "sbert" or "sbert" in self.retrieval_method or "sentence" in self.retrieval_method:
            self.encoder = SBERTEncoder(
                model_name=self.retrieval_method,
                model_path=config.retrieval_model_path,
                max_length=config.retrieval_query_max_length,
                use_fp16=config.retrieval_use_fp16
            )
        else:
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=config.retrieval_model_path,
                pooling_method=config.retrieval_pooling_method,
                max_length=config.retrieval_query_max_length,
                use_fp16=config.retrieval_use_fp16
            )
        
        # Initialize reranker if specified
        self.reranker = None
        if hasattr(config, 'reranker_model_path') and config.reranker_model_path:
            self.reranker = BGEReranker(
                model_path=config.reranker_model_path,
                use_fp16=config.retrieval_use_fp16,
                max_length=getattr(config, 'reranker_max_length', 512)
            )
        
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
            
        # Get initial retrieval results
        initial_num = num * 2 if self.reranker else num  # Retrieve more for reranking
        
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=initial_num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        
        # Apply reranking if available
        if self.reranker and len(results) > 0:
            results = self.reranker.rerank(query, results, top_k=num)
            # Recompute scores after reranking (optional)
            scores = scores[:len(results)]  # Adjust scores array
        
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        
        # Get initial retrieval results
        initial_num = num * 2 if self.reranker else num
        
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=initial_num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # Load docs
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i*initial_num : (i+1)*initial_num] for i in range(len(batch_idxs))]
            
            # Apply reranking for each query in batch
            if self.reranker:
                reranked_batch_results = []
                for i, (query, query_results) in enumerate(zip(query_batch, batch_results)):
                    if len(query_results) > 0:
                        reranked_results = self.reranker.rerank(query, query_results, top_k=num)
                        reranked_batch_results.append(reranked_results)
                        # Adjust scores
                        batch_scores[i] = batch_scores[i][:len(reranked_results)]
                    else:
                        reranked_batch_results.append(query_results)
                batch_results = reranked_batch_results
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results
        
def get_retriever(config):
    """Factory function to create appropriate retriever based on config"""
    if config.retrieval_method.lower() == "bm25":
        return BM25Retriever(config)
    elif config.retrieval_method.lower() in ["openai", "bge", "e5", "sbert", "sentence"] or any(name in config.retrieval_method.lower() for name in ["openai", "bge", "e5", "sbert", "sentence"]):
        return DenseRetriever(config)
    else:
        # Default to dense retriever for other methods
        return DenseRetriever(config)
    
#####################################
# FastAPI server below
#####################################

class Config:
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        openai_api_key: str = None,
        openai_base_url: str = None,
        reranker_model_path: str = None,  # Add reranker support
        reranker_max_length: int = 512
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.reranker_model_path = reranker_model_path
        self.reranker_max_length = reranker_max_length


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.get("/info")
def get_server_info():
    """Get retrieval server information"""
    try:
        retriever_info = {
            "retriever_type": config.retrieval_method,
            "model_name": config.retrieval_model_path,
            "index_path": config.index_path,
            "corpus_path": config.corpus_path,
            "topk": config.retrieval_topk,
            "faiss_gpu": config.faiss_gpu,
            "server_responsive": True
        }
        
        # Add index type information
        if "bm25" in config.retrieval_method.lower():
            retriever_info["index_type"] = "BM25"
        elif config.index_path:
            if "Flat" in config.index_path:
                retriever_info["index_type"] = "FAISS_Flat"
            elif "HNSW" in config.index_path:
                retriever_info["index_type"] = "FAISS_HNSW"
            else:
                retriever_info["index_type"] = "FAISS_Unknown"
        else:
            retriever_info["index_type"] = "Unknown"
        
        # Add reranker information if available
        if hasattr(config, 'reranker_model_path') and config.reranker_model_path:
            retriever_info["reranker_model"] = config.reranker_model_path
            retriever_info["has_reranker"] = True
        else:
            retriever_info["has_reranker"] = False
            
        return retriever_info
        
    except Exception as e:
        return {
            "error": str(e),
            "server_responsive": True,
            "retriever_type": "error",
            "model_name": "error",
            "index_type": "error"
        }

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    retrieval_result = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    # Handle different return formats
    if request.return_scores:
        # When return_score=True, expect (results, scores) tuple
        if isinstance(retrieval_result, tuple) and len(retrieval_result) == 2:
            results, scores = retrieval_result
        else:
            # Fallback: results without scores
            results = retrieval_result
            scores = [[0.0] * len(res) for res in results]
    else:
        # When return_score=False, expect just results
        if isinstance(retrieval_result, tuple):
            results = retrieval_result[0]  # Take first element if tuple
        else:
            results = retrieval_result
        scores = None
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores and scores:
            # If scores are returned, combine them with results
            combined = []
            for j, doc in enumerate(single_result):
                score = scores[i][j] if j < len(scores[i]) else 0.0
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            # Just return documents without scores
            formatted_docs = []
            for doc in single_result:
                formatted_docs.append({"document": doc})
            resp.append(formatted_docs)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API key')
    parser.add_argument('--openai_base_url', type=str, default=None, help='OpenAI base URL')
    parser.add_argument('--reranker_model', type=str, default=None, help='BGE reranker model path')  # Add reranker argument

    args = parser.parse_args()
    
    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        reranker_model_path=args.reranker_model,  # Add reranker config
        reranker_max_length=512
    )

    retriever = get_retriever(config)
    uvicorn.run(app, host="0.0.0.0", port=8000)
