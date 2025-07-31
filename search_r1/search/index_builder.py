import os
import faiss
import json
import warnings
import numpy as np
from typing import cast, List, Dict
import shutil
import subprocess
import argparse
import torch
from tqdm import tqdm
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
import openai  
import tiktoken
import time
import requests

class OpenAIEmbedder:
    """Wrapper class for OpenAI Embedding API"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str = None):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url  # Support custom base_url, e.g., Azure OpenAI
        )
        self.model = model
        self.max_tokens = 8191  # Maximum token limit for text-embedding-3-small
        
        # Initialize tiktoken encoder for accurate token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # If the model is not supported, use the default encoder
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
    def _truncate_text(self, text: str) -> str:
        """Accurately truncate text to the maximum token limit"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.max_tokens:
            return text
        
        # Truncate to the maximum number of tokens
        truncated_tokens = tokens[:self.max_tokens]
        return self.encoding.decode(truncated_tokens)
        
    def encode_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Batch encode texts into vectors"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI Embedding"):
            batch_texts = texts[i:i + batch_size]
            
            # Accurately truncate texts
            truncated_texts = []
            for text in batch_texts:
                truncated_text = self._truncate_text(text)
                truncated_texts.append(truncated_text)
            
            # Check total token count in the batch
            total_tokens = sum(len(self.encoding.encode(text)) for text in truncated_texts)
            
            # If the batch is too large, reduce batch size
            if total_tokens > self.max_tokens * len(truncated_texts) * 0.8:  # Conservative estimate
                # Process one by one to avoid oversized batch
                for single_text in truncated_texts:
                    single_batch_result = self._encode_single_batch([single_text])
                    all_embeddings.extend(single_batch_result)
            else:
                batch_result = self._encode_single_batch(truncated_texts)
                all_embeddings.extend(batch_result)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def _encode_single_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a single batch"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                
                # Extract embedding vectors
                embeddings = [data.embedding for data in response.data]
                return embeddings
                
            except openai.BadRequestError as e:
                if "maximum context length" in str(e):
                    # If still too long, further truncate
                    print(f"Further truncating texts due to length error: {e}")
                    texts = [self._truncate_text(text, max_tokens=self.max_tokens // 2) for text in texts]
                    retry_count += 1
                else:
                    raise
                    
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
                
        raise Exception("Failed to encode after maximum retries")
    
    def _truncate_text(self, text: str, max_tokens: int = None) -> str:
        """Accurately truncate text to the specified token limit"""
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate to the specified number of tokens
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

class BGEEmbedder:
    """BGE Embedding Model Wrapper"""
    
    def __init__(self, model_path: str, use_fp16: bool = False, max_length: int = 512):
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        
        # Load BGE model and tokenizer
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.model.eval()
        self.model.cuda()
        if use_fp16:
            self.model = self.model.half()
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
        """Batch encode texts using BGE model"""
        all_embeddings = []
        
        # Add BGE instruction prefix for passages (not for queries in indexing)
        if not is_query:
            # For BGE, we don't add special prefixes during indexing
            processed_texts = texts
        else:
            # For queries, we would add instruction, but this is handled in retrieval
            processed_texts = texts
        
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="BGE Embedding"):
            batch_texts = processed_texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length
            ).to('cuda')
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token for BGE
                embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

def load_model(
        model_path: str, 
        use_fp16: bool = False
    ):
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


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
            'json', 
            data_files=corpus_path,
            split="train",
            num_proc=4)
    return corpus


class Index_Builder:
    def __init__(
            self, 
            retrieval_method,
            model_path,
            corpus_path,
            save_dir,
            max_length,
            batch_size,
            use_fp16,
            pooling_method,
            faiss_type=None,
            embedding_path=None,
            save_embedding=False,
            faiss_gpu=False,
            openai_api_key=None,
            openai_base_url=None,
            openai_model="text-embedding-3-small"
        ):
        
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else 'Flat'
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        
        # OpenAI related parameters
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model

        self.gpu_num = torch.cuda.device_count()
        # Prepare save directory
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")
        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")
        self.corpus = load_corpus(self.corpus_path)
        print("Finish loading...")

    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the directory path exists and if there is content."""
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        r"""Construct different indexes based on the selected retrieval method."""
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """Build BM25 index using the Pyserini library.

        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """

        # To use the pyserini pipeline, we first need to place the jsonl file in the folder 
        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)

        # Copy the corpus file to the temp directory
        shutil.copyfile(self.corpus_path, temp_file_path)
        
        print("Start building bm25 index...")
        pyserini_args = ["--collection", "JsonCollection",
                         "--input", temp_dir,
                         "--index", self.save_dir,
                         "--generator", "DefaultLuceneDocumentGenerator",
                         "--threads", "1"]
       
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)

        shutil.rmtree(temp_dir)
        
        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(
                embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(corpus_size, hidden_size)
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype
        )
        length = all_embeddings.shape[0]
        # Save in batches
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i: j] = all_embeddings[i: j]
        else:
            memmap[:] = all_embeddings

    def encode_all(self):
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.batch_size = self.batch_size * self.gpu_num

        all_embeddings = []

        for start_idx in tqdm(range(0, len(self.corpus), self.batch_size), desc='Inference Embeddings:'):
            # batch_data = ['"' + title + '"\n' + text for title, text in zip(batch_data_title, batch_data_text)]
            batch_data = self.corpus[start_idx:start_idx+self.batch_size]['contents']

            if self.retrieval_method == "e5":
                batch_data = [f"passage: {doc}" for doc in batch_data]

            inputs = self.tokenizer(
                        batch_data,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length,
            ).to('cuda')

            inputs = {k: v.cuda() for k, v in inputs.items()}

            # TODO: support encoder-only T5 model
            if "T5" in type(self.encoder).__name__:
                # T5-based retrieval model
                decoder_input_ids = torch.zeros(
                    (inputs['input_ids'].shape[0], 1), dtype=torch.long
                ).to(inputs['input_ids'].device)
                output = self.encoder(
                    **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
                )
                embeddings = output.last_hidden_state[:, 0, :]

            else:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(output.pooler_output, 
                                    output.last_hidden_state, 
                                    inputs['attention_mask'],
                                    self.pooling_method)
                if  "dpr" not in self.retrieval_method:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            embeddings = cast(torch.Tensor, embeddings)
            embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    def encode_all_openai(self):
        """Encode all documents using the OpenAI API"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI embedding")
        
        embedder = OpenAIEmbedder(
            api_key=self.openai_api_key,
            model=self.openai_model,
            base_url=self.openai_base_url
        )
        
        # Extract document contents
        all_texts = []
        for item in tqdm(self.corpus, desc="Preparing texts"):
            text = item['contents']
            all_texts.append(text)
        
        # Batch encode using OpenAI API
        all_embeddings = embedder.encode_batch(all_texts, batch_size=self.batch_size)
        
        return all_embeddings

    def encode_all_bge(self):
        """Encode all documents using BGE model"""
        embedder = BGEEmbedder(
            model_path=self.model_path,
            use_fp16=self.use_fp16,
            max_length=self.max_length
        )
        
        # Extract document contents
        all_texts = []
        for item in tqdm(self.corpus, desc="Preparing texts"):
            text = item['contents']
            all_texts.append(text)
        
        # Batch encode using BGE
        all_embeddings = embedder.encode_batch(all_texts, batch_size=self.batch_size, is_query=False)
        
        return all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        """Build dense index, supporting OpenAI embedding and BGE"""
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")
        
        # Check embedding method
        if self.retrieval_method == "openai" or "openai" in self.retrieval_method:
            # OpenAI embedding logic (existing code)
            if self.embedding_path is not None:
                corpus_size = len(self.corpus)
                hidden_size = 1536  
                all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
            else:
                all_embeddings = self.encode_all_openai()
                if self.save_embedding:
                    self._save_embedding(all_embeddings)
                del self.corpus
                
        elif self.retrieval_method == "bge" or "bge" in self.retrieval_method:
            # BGE embedding logic
            if self.embedding_path is not None:
                # For BGE models, get hidden size from config
                config = AutoConfig.from_pretrained(self.model_path)
                hidden_size = config.hidden_size
                corpus_size = len(self.corpus)
                all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
            else:
                all_embeddings = self.encode_all_bge()
                if self.save_embedding:
                    self._save_embedding(all_embeddings)
                del self.corpus
                
        else:
            # Use the original local model encoding logic
            self.encoder, self.tokenizer = load_model(model_path=self.model_path, 
                                                      use_fp16=self.use_fp16)
            if self.embedding_path is not None:
                hidden_size = self.encoder.config.hidden_size
                corpus_size = len(self.corpus)
                all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
            else:
                all_embeddings = self.encode_all()
                if self.save_embedding:
                    self._save_embedding(all_embeddings)
                del self.corpus

        # Build faiss index (existing code)
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)
        
        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)

        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")


MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    'jina': 'mean'
}


def main():
    parser = argparse.ArgumentParser(description="Creating index.")

    # Basic parameters
    parser.add_argument('--retrieval_method', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--save_dir', default='indexes/', type=str)

    # Parameters for building dense index
    parser.add_argument('--max_length', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--use_fp16', default=False, action='store_true')
    parser.add_argument('--pooling_method', type=str, default=None)
    parser.add_argument('--faiss_type', default=None, type=str)
    parser.add_argument('--embedding_path', default=None, type=str)
    parser.add_argument('--save_embedding', action='store_true', default=False)
    parser.add_argument('--faiss_gpu', default=False, action='store_true')
    
    # OpenAI parameters
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API key')
    parser.add_argument('--openai_base_url', type=str, default=None, help='OpenAI base URL (for Azure OpenAI)')
    parser.add_argument('--openai_model', type=str, default='text-embedding-3-small', help='OpenAI embedding model')
    
    args = parser.parse_args()

    if args.pooling_method is None:
        pooling_method = 'mean'
        for k, v in MODEL2POOLING.items():
            if k in args.retrieval_method.lower():
                pooling_method = v
                break
    else:
        if args.pooling_method not in ['mean', 'cls', 'pooler']:
            raise NotImplementedError
        else:
            pooling_method = args.pooling_method

    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=pooling_method,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()