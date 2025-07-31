import json
import os
import re
import ast
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Will use simple text matching instead.")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Will use simple text matching instead.")

from inference_engine import search_r1_inference, get_or_create_model


class ECHRTestEvaluator:
    """Evaluator for ECHR QA testing with guide matching and similarity scoring"""
    
    def __init__(self, 
                 echr_qa_path: str = "data/echr_qa/echr_qa_compact.json",
                 echr_guide_path: str = "data/echr_guide.jsonl",
                 results_dir: str = "test_results",
                 preload_model: bool = True,
                 model_id: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3",
                 use_quantization: bool = True,
                 quantization_bits: int = 8,
                 evaluation_topk: int = 200):
        
        self.echr_qa_path = echr_qa_path
        self.echr_guide_path = echr_guide_path
        self.results_dir = results_dir
        self.preload_model = preload_model
        self.model_id = model_id
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.evaluation_topk = evaluation_topk  # 评估时使用的topk，用于查找目标段落排名
        
        # Create organized directory structure
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "latest"), exist_ok=True)
        
        # Load data
        self.qa_data = self._load_qa_data()
        self.guide_data = self._load_guide_data()
        
        # Initialize similarity model
        self.similarity_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded sentence-transformers similarity model")
            except Exception as e:
                print(f"Failed to load sentence-transformers: {e}")
        
        # Initialize BERT score model (alternative)
        self.bert_model = None
        self.bert_tokenizer = None
        if TRANSFORMERS_AVAILABLE and not self.similarity_model:
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
                print("Loaded BERT model for similarity")
            except Exception as e:
                print(f"Failed to load BERT model: {e}")
        
        # Preload SearchR1 model if requested
        self.search_model = None
        if self.preload_model:
            print(f"🔄 Preloading SearchR1 model: {self.model_id}")
            try:
                self.search_model = get_or_create_model(
                    model_id=self.model_id,
                    use_quantization=self.use_quantization,
                    quantization_bits=self.quantization_bits
                )
                print(f"✅ SearchR1 model preloaded successfully")
            except Exception as e:
                print(f"❌ Failed to preload SearchR1 model: {e}")
                self.preload_model = False
    
    def _load_qa_data(self) -> List[Dict]:
        """Load ECHR QA data"""
        with open(self.echr_qa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _get_retriever_info(self, retrieval_server_url: str = "http://127.0.0.1:8000") -> Dict:
        """Get retriever information from the retrieval server"""
        try:
            # Try to get server info endpoint
            info_response = requests.get(f"{retrieval_server_url}/info", timeout=5)
            if info_response.status_code == 200:
                return info_response.json()
        except:
            pass
        
        try:
            # Fallback: test with a simple query to see if server is responsive
            test_payload = {
                "queries": ["test"],
                "topk": 1,
                "return_scores": False
            }
            test_response = requests.post(f"{retrieval_server_url}/retrieve", json=test_payload, timeout=10)
            if test_response.status_code == 200:
                # Server is responsive, but we don't have detailed info
                # We'll try to infer from the response or return basic info
                return {
                    "retriever_type": "unknown",
                    "model_name": "unknown", 
                    "index_type": "unknown",
                    "server_responsive": True,
                    "server_url": retrieval_server_url
                }
        except Exception as e:
            print(f"Failed to get retriever info: {e}")
        
        return {
            "retriever_type": "unknown",
            "model_name": "unknown",
            "index_type": "unknown", 
            "server_responsive": False,
            "server_url": retrieval_server_url,
            "error": "Could not connect to retrieval server"
        }
    
    def _load_guide_data(self) -> Dict[str, Dict]:
        """Load ECHR guide data"""
        guide_data = {}
        with open(self.echr_guide_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                guide_data[item['id']] = item
        return guide_data
    
    def _extract_guide_info(self, content: str) -> Tuple[str, str, str]:
        """Extract guide_id, paragraph_id, and paragraph from content"""
        try:
            # Parse content: "guide id: guide_art_1_eng; paragraph id: 1; paragraph: ..."
            parts = content.split('; ')
            guide_id = parts[0].replace('guide id: ', '').strip()
            paragraph_id = parts[1].replace('paragraph id: ', '').strip()
            paragraph = parts[2].replace('paragraph: ', '').strip()
            return guide_id, paragraph_id, paragraph
        except:
            return "", "", content
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        if self.similarity_model:
            try:
                embeddings = self.similarity_model.encode([text1, text2])
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                return similarity
            except Exception as e:
                print(f"Similarity computation error: {e}")
                return self._simple_text_similarity(text1, text2)
        
        elif self.bert_model and self.bert_tokenizer:
            try:
                return self._bert_similarity(text1, text2)
            except Exception as e:
                print(f"BERT similarity error: {e}")
                return self._simple_text_similarity(text1, text2)
        
        else:
            return self._simple_text_similarity(text1, text2)
    
    def _bert_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using BERT embeddings"""
        def get_bert_embedding(text):
            inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            return embedding
        
        emb1 = get_bert_embedding(text1)
        emb2 = get_bert_embedding(text2)
        
        # Cosine similarity
        similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return similarity
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _get_ground_truth_guide_text(self, guide_id: str, paragraph_ids: List[int]) -> str:
        """Get ground truth guide text from guide data based on guide_id and paragraph_ids"""
        if not guide_id or not paragraph_ids:
            return ""
        
        guide_texts = []
        for paragraph_id in paragraph_ids:
            # Find the corresponding guide text in guide_data
            for doc_id, doc_content in self.guide_data.items():
                content = doc_content.get('contents', '')
                doc_guide_id, doc_paragraph_id, paragraph_text = self._extract_guide_info(content)
                
                # Match both guide_id and paragraph_id
                if doc_guide_id == guide_id and doc_paragraph_id == str(paragraph_id):
                    guide_texts.append(paragraph_text)
                    break
        
        # Join all found guide texts
        return " ".join(guide_texts)
    
    def _check_guide_retrieval(self, search_queries: List[str], target_guide_id: Optional[str] = None, target_paragraphs: Optional[List[int]] = None) -> Dict:
        """Check if retrieval found relevant guide information using actual retrieval API"""
        
        # First, try to get actual retrieval results
        actual_retrieval_score = 0.0
        found_paragraphs = []
        found_guides = []
        correctly_matched_docs = []  # Documents with both correct guide and paragraph
        total_docs = 0
        
        # Track ranking information for target paragraphs
        target_paragraph_ranks = {}  # Maps paragraph_id to its best rank position
        
        try:
            # Call actual retrieval API with larger topk for evaluation
            payload = {
                "queries": search_queries,
                "topk": self.evaluation_topk,  # 使用评估专用的topk
                "return_scores": True
            }
            response = requests.post("http://127.0.0.1:8000/retrieve", json=payload, timeout=10)
            
            if response.status_code == 200:
                results = response.json()
                total_docs = sum(len(query_results) for query_results in results.get('result', []))
                
                # Analyze retrieved documents
                for i, query_results in enumerate(results.get('result', [])):
                    for rank, doc_info in enumerate(query_results):
                        doc_content = doc_info.get('document', {}).get('contents', '')
                        
                        # Extract guide info from retrieved content
                        guide_id, paragraph_id, paragraph_text = self._extract_guide_info(doc_content)
                        
                        if guide_id:
                            found_guides.append(guide_id)
                            
                        if paragraph_id and paragraph_id.isdigit():
                            found_paragraphs.append(int(paragraph_id))
                            
                        # Track ranking of target paragraphs
                        if target_guide_id and target_paragraphs:
                            # Check if this document has the correct guide
                            if guide_id == target_guide_id and paragraph_id and paragraph_id.isdigit():
                                paragraph_id_int = int(paragraph_id)
                                # If this is one of our target paragraphs and we haven't recorded its rank yet,
                                # or if this is a better rank (lower number), record it
                                if paragraph_id_int in target_paragraphs:
                                    if paragraph_id_int not in target_paragraph_ranks or rank < target_paragraph_ranks[paragraph_id_int]:
                                        target_paragraph_ranks[paragraph_id_int] = rank + 1  # 1-indexed ranking
                            
                            # Check if this document has BOTH correct guide and correct paragraph
                            if (guide_id == target_guide_id and 
                                paragraph_id and paragraph_id.isdigit() and 
                                int(paragraph_id) in target_paragraphs):
                                correctly_matched_docs.append({
                                    'guide_id': guide_id,
                                    'paragraph_id': int(paragraph_id),
                                    'content': paragraph_text,
                                    'rank': rank + 1  # 1-indexed ranking
                                })
                
                # Calculate retrieval score based on ground truth - require BOTH guide and paragraph to match
                if target_guide_id and target_paragraphs:
                    # Count correctly matched paragraphs (those with both correct guide and paragraph)
                    correctly_matched_paragraphs = [doc['paragraph_id'] for doc in correctly_matched_docs]
                    correct_paragraph_set = set(correctly_matched_paragraphs)
                    target_paragraph_set = set(target_paragraphs)
                    
                    if target_paragraph_set:
                        # Calculate precision, recall, F1 based on correctly matched documents only
                        paragraph_overlap = len(correct_paragraph_set.intersection(target_paragraph_set))
                        paragraph_recall = paragraph_overlap / len(target_paragraph_set)
                        paragraph_precision = paragraph_overlap / len(correctly_matched_paragraphs) if correctly_matched_paragraphs else 0.0
                        paragraph_f1 = (2 * paragraph_recall * paragraph_precision) / (paragraph_recall + paragraph_precision) if (paragraph_recall + paragraph_precision) > 0 else 0.0
                        
                        # Use F1 score as the retrieval score (requiring both guide and paragraph to match)
                        actual_retrieval_score = paragraph_f1
                    else:
                        actual_retrieval_score = 0.0
                
        except Exception as e:
            print(f"Error: Retrieval server is not available - {e}")
            # Return error result instead of falling back to keyword matching
            return {
                'retrieval_score': 0.0,
                'matched_guides': [],
                'found_paragraphs': [],
                'correctly_matched_docs': [],
                'correctly_matched_paragraphs': [],
                'target_guide': target_guide_id,
                'target_paragraphs': target_paragraphs,
                'target_found': False,
                'paragraph_overlap': 0,
                'correct_paragraph_overlap': 0,
                'total_retrieved_docs': 0,
                'target_paragraph_ranks': {},
                'error': f"Retrieval server unavailable: {e}"
            }
        
        return {
            'retrieval_score': actual_retrieval_score,
            'matched_guides': list(set(found_guides)),
            'found_paragraphs': list(set(found_paragraphs)),
            'correctly_matched_docs': correctly_matched_docs,
            'correctly_matched_paragraphs': list(set([doc['paragraph_id'] for doc in correctly_matched_docs])),
            'target_guide': target_guide_id,
            'target_paragraphs': target_paragraphs,
            'target_found': target_guide_id in found_guides if target_guide_id else False,
            'paragraph_overlap': len(set(found_paragraphs).intersection(set(target_paragraphs))) if target_paragraphs else 0,
            'correct_paragraph_overlap': len(set([doc['paragraph_id'] for doc in correctly_matched_docs]).intersection(set(target_paragraphs))) if target_paragraphs else 0,
            'total_retrieved_docs': total_docs,
            'target_paragraph_ranks': target_paragraph_ranks
        }

    def _extract_answer_from_response(self, response: str) -> str:
        """Extract the final answer from model response"""
        # Look for content within <answer> tags
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        matches = answer_pattern.findall(response)
        
        if matches:
            return matches[-1].strip()
        
        # Fallback: return the last substantial paragraph
        lines = response.strip().split('\n')
        for line in reversed(lines):
            if line.strip() and len(line.strip()) > 50:
                return line.strip()
        
        return response.strip()
    
    def evaluate_single_qa(self, 
                          qa_item: Dict, 
                          index: int,
                          model_id: str,
                          topk: int = 15) -> Dict:
        """Evaluate a single QA item"""
        
        print(f"\n=== Evaluating QA Item {index + 1} ===")
        
        # Extract ground truth information
        # TODO: think of a nother way compute the score of model output
        expected_answer = qa_item['answer_no_citations']
        target_guide_id = qa_item.get('guide', None)
        
        # Parse target paragraphs from string format "[2, 3, 4, 5, 8, 12]"
        target_paragraphs = None
        if 'paragraphs' in qa_item and qa_item['paragraphs']:
            try:
                target_paragraphs = ast.literal_eval(qa_item['paragraphs'])
                if not isinstance(target_paragraphs, list):
                    target_paragraphs = None
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse paragraphs: {qa_item['paragraphs']}")
                target_paragraphs = None
        
        # Generate a question (if not provided, skip this item)
        # TODO: check if this works
        if 'question' in qa_item and qa_item['question']:
            question = qa_item['question']
            question += f"Relevant Reference Cases: {qa_item['citations']}"
        else:
            # Skip items without questions
            print(f"Warning: No question found for item {index + 1}, skipping...")
            return {
                'index': index,
                'question': "",
                'expected_answer': expected_answer,
                'model_answer': "",
                'clean_answer': "",
                'similarity_score': 0.0,
                'ground_truth_guide_text': "",
                'guide_similarity_score': 0.0,
                'search_count': 0,
                'search_queries': [],
                'retrieval_score': 0.0,
                'matched_guides': [],
                'found_paragraphs': [],
                'correctly_matched_docs': [],
                'correctly_matched_paragraphs': [],
                'target_guide': target_guide_id,
                'target_paragraphs': target_paragraphs,
                'paragraph_overlap': 0,
                'correct_paragraph_overlap': 0,
                'total_retrieved_docs': 0,
                'target_paragraph_ranks': {},
                'log_path': "",
                'duration': 0,
                'success': False,
                'error': "No question provided"
            }
        
        print(f"Question: {question}")
        print(f"Expected answer length: {len(expected_answer)} chars")
        print(f"Target guide: {target_guide_id}")
        print(f"Target paragraphs: {target_paragraphs}")
        
        try:
            # Run inference using preloaded model or standard function
            if self.preload_model and self.search_model:
                # Use preloaded model
                model_answer, log_path, metadata = self.search_model.inference(
                    question=question,
                    log_dir=os.path.join(self.results_dir, "logs"),
                    topk=topk
                )
            else:
                # Fall back to standard function
                model_answer, log_path, metadata = search_r1_inference(
                    question=question,
                    model_id=model_id,
                    log_dir=os.path.join(self.results_dir, "logs"),
                    topk=topk
                )
            
            # Extract clean answer
            clean_answer = self._extract_answer_from_response(model_answer)
            
            # Compute similarity with expected answer
            similarity_score = self._compute_similarity(clean_answer, expected_answer)
            
            # Get ground truth guide text and compute similarity
            ground_truth_guide_text = ""
            guide_similarity_score = 0.0
            if target_guide_id and target_paragraphs:
                ground_truth_guide_text = self._get_ground_truth_guide_text(target_guide_id, target_paragraphs)
                if ground_truth_guide_text.strip():
                    guide_similarity_score = self._compute_similarity(clean_answer, ground_truth_guide_text)
            
            # Check retrieval effectiveness using ground truth
            retrieval_check = self._check_guide_retrieval(
                metadata.get('search_queries', []),
                target_guide_id=target_guide_id,
                target_paragraphs=target_paragraphs
            )
            
            result = {
                'index': index,
                'question': question,
                'expected_answer': expected_answer,
                'model_answer': model_answer,
                'clean_answer': clean_answer,
                'similarity_score': similarity_score,
                'ground_truth_guide_text': ground_truth_guide_text,
                'guide_similarity_score': guide_similarity_score,
                'search_count': metadata.get('search_count', 0),
                'search_queries': metadata.get('search_queries', []),
                'retrieval_score': retrieval_check['retrieval_score'],
                'matched_guides': retrieval_check['matched_guides'],
                'found_paragraphs': retrieval_check.get('found_paragraphs', []),
                'correctly_matched_docs': retrieval_check.get('correctly_matched_docs', []),
                'correctly_matched_paragraphs': retrieval_check.get('correctly_matched_paragraphs', []),
                'target_guide': target_guide_id,
                'target_paragraphs': target_paragraphs,
                'paragraph_overlap': retrieval_check.get('paragraph_overlap', 0),
                'correct_paragraph_overlap': retrieval_check.get('correct_paragraph_overlap', 0),
                'total_retrieved_docs': retrieval_check.get('total_retrieved_docs', 0),
                'target_paragraph_ranks': retrieval_check.get('target_paragraph_ranks', {}),
                'log_path': log_path,
                'duration': metadata.get('duration', 0),
                'success': True,
                'error': None
            }
            
            print(f"Similarity Score: {similarity_score:.3f}")
            print(f"Guide Similarity Score: {guide_similarity_score:.3f}")
            print(f"Ground Truth Guide Text Length: {len(ground_truth_guide_text)} chars")
            print(f"Search Count: {metadata.get('search_count', 0)}")
            print(f"Retrieval Score: {retrieval_check['retrieval_score']:.3f}")
            print(f"Correct Match Overlap: {retrieval_check.get('correct_paragraph_overlap', 0)}/{len(target_paragraphs) if target_paragraphs else 0}")
            print(f"Correctly Matched Documents: {len(retrieval_check.get('correctly_matched_docs', []))}")
            
            # Print ranking information
            target_paragraph_ranks = retrieval_check.get('target_paragraph_ranks', {})
            if target_paragraph_ranks:
                avg_rank = np.mean(list(target_paragraph_ranks.values()))
                print(f"Target Paragraphs Found: {len(target_paragraph_ranks)}/{len(target_paragraphs) if target_paragraphs else 0}")
                print(f"Average Rank of Found Paragraphs: {avg_rank:.1f}")
                print(f"Top-15 Found: {sum(1 for rank in target_paragraph_ranks.values() if rank <= 15)}/{len(target_paragraphs) if target_paragraphs else 0}")
                print(f"Ranks: {target_paragraph_ranks}")
            else:
                print(f"Target Paragraphs Found: 0/{len(target_paragraphs) if target_paragraphs else 0}")
            
            return result
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            result = {
                'index': index,
                'question': question,
                'expected_answer': expected_answer,
                'model_answer': "",
                'clean_answer': "",
                'similarity_score': 0.0,
                'ground_truth_guide_text': "",
                'guide_similarity_score': 0.0,
                'search_count': 0,
                'search_queries': [],
                'retrieval_score': 0.0,
                'matched_guides': [],
                'found_paragraphs': [],
                'correctly_matched_docs': [],
                'correctly_matched_paragraphs': [],
                'target_guide': target_guide_id,
                'target_paragraphs': target_paragraphs,
                'paragraph_overlap': 0,
                'correct_paragraph_overlap': 0,
                'total_retrieved_docs': 0,
                'target_paragraph_ranks': {},
                'log_path': "",
                'duration': 0,
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def run_evaluation(self, 
                      model_id: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3",
                      max_samples: int = 5,
                      topk: int = 15,
                      retrieval_server_url: str = "http://127.0.0.1:8000") -> Dict:
        """Run full evaluation on ECHR QA dataset"""
        
        print(f"=== Starting ECHR QA Evaluation ===")
        print(f"Model: {model_id}")
        print(f"Max samples: {max_samples}")
        print(f"TopK for model: {topk}")
        print(f"TopK for evaluation: {self.evaluation_topk}")
        print(f"Results directory: {self.results_dir}")
        
        # Create run-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.join(self.results_dir, "runs", f"run_{timestamp}")
        os.makedirs(self.current_run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.current_run_dir, "intermediate"), exist_ok=True)
        
        print(f"Run directory: {self.current_run_dir}")
        
        # Get retriever information
        print("Getting retriever information...")
        retriever_info = self._get_retriever_info(retrieval_server_url)
        print(f"Retriever info: {retriever_info}")
        
        # Sample data
        sample_data = self.qa_data[:max_samples]
        
        results = []
        
        for i, qa_item in enumerate(tqdm(sample_data, desc="Evaluating QA items")):
            result = self.evaluate_single_qa(qa_item, i, model_id, topk)
            results.append(result)
            
            # Save intermediate results
            if (i + 1) % 2 == 0:
                self._save_intermediate_results(results, i + 1)
        
        # Compute summary statistics
        summary = self._compute_summary(results)
        
        # Save final results with retriever info
        self._save_final_results(results, summary, model_id, max_samples, topk, retriever_info)
        
        return {
            'results': results,
            'summary': summary,
            'retriever_info': retriever_info
        }
    
    def _save_intermediate_results(self, results: List[Dict], count: int):
        """Save intermediate results"""
        if not hasattr(self, 'current_run_dir'):
            # Fallback if current_run_dir is not set
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_run_dir = os.path.join(self.results_dir, "runs", f"run_{timestamp}")
            os.makedirs(os.path.join(self.current_run_dir, "intermediate"), exist_ok=True)
        
        intermediate_file = os.path.join(self.current_run_dir, "intermediate", f"step_{count:03d}.json")
        
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Saved intermediate results to {intermediate_file}")
    
    def _compute_summary(self, results: List[Dict]) -> Dict:
        """Compute summary statistics"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                'total_samples': len(results),
                'successful_samples': 0,
                'success_rate': 0.0,
                'avg_similarity': 0.0,
                'avg_search_count': 0.0,
                'avg_retrieval_score': 0.0,
                'avg_duration': 0.0,
                'avg_correct_paragraph_overlap': 0.0,
                'guide_found_rate': 0.0,
                'correct_match_rate': 0.0,
                'avg_retrieved_docs': 0.0,
                'avg_correctly_matched_docs': 0.0
            }
        
        avg_similarity = np.mean([r['similarity_score'] for r in successful_results])
        avg_guide_similarity = np.mean([r['guide_similarity_score'] for r in successful_results])
        avg_search_count = np.mean([r['search_count'] for r in successful_results])
        avg_retrieval_score = np.mean([r['retrieval_score'] for r in successful_results])
        avg_duration = np.mean([r['duration'] for r in successful_results])
        
        # Correct matching metrics (both guide and paragraph must match)
        correct_paragraph_overlaps = [r.get('correct_paragraph_overlap', 0) for r in successful_results]
        avg_correct_paragraph_overlap = np.mean(correct_paragraph_overlaps) if correct_paragraph_overlaps else 0.0
        
        # Guide finding rate (any guide found)
        guide_matches = [1 if r.get('target_guide') in r.get('matched_guides', []) else 0 for r in successful_results if r.get('target_guide')]
        guide_found_rate = np.mean(guide_matches) if guide_matches else 0.0
        
        # Correct match rate (documents with both correct guide and correct paragraph)
        correct_matches = [1 if r.get('correct_paragraph_overlap', 0) > 0 else 0 for r in successful_results]
        correct_match_rate = np.mean(correct_matches) if correct_matches else 0.0
        
        # Retrieved documents count
        retrieved_docs = [r.get('total_retrieved_docs', 0) for r in successful_results]
        avg_retrieved_docs = np.mean(retrieved_docs) if retrieved_docs else 0.0
        
        # Correctly matched documents count
        correctly_matched_docs = [len(r.get('correctly_matched_docs', [])) for r in successful_results]
        avg_correctly_matched_docs = np.mean(correctly_matched_docs) if correctly_matched_docs else 0.0
        
        # Ranking statistics for target paragraphs
        all_target_ranks = []
        target_paragraphs_found = 0
        total_target_paragraphs = 0
        
        for r in successful_results:
            target_paragraph_ranks = r.get('target_paragraph_ranks', {})
            target_paragraphs = r.get('target_paragraphs', [])
            
            # Count how many target paragraphs were found
            target_paragraphs_found += len(target_paragraph_ranks)
            total_target_paragraphs += len(target_paragraphs)
            
            # Collect all ranking positions
            all_target_ranks.extend(target_paragraph_ranks.values())
        
        # Calculate average rank of found target paragraphs
        avg_target_rank = np.mean(all_target_ranks) if all_target_ranks else 0.0
        
        # Calculate percentage of target paragraphs found
        target_paragraphs_found_rate = target_paragraphs_found / total_target_paragraphs if total_target_paragraphs > 0 else 0.0
        
        # Calculate percentage of target paragraphs found in top 15
        top15_found = sum(1 for rank in all_target_ranks if rank <= 15)
        top15_found_rate = top15_found / total_target_paragraphs if total_target_paragraphs > 0 else 0.0
        
        return {
            'total_samples': len(results),
            'successful_samples': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_similarity': float(avg_similarity),
            'avg_guide_similarity': float(avg_guide_similarity),
            'avg_search_count': float(avg_search_count),
            'avg_retrieval_score': float(avg_retrieval_score),
            'avg_duration': float(avg_duration),
            'avg_correct_paragraph_overlap': float(avg_correct_paragraph_overlap),
            'guide_found_rate': float(guide_found_rate),
            'correct_match_rate': float(correct_match_rate),
            'avg_retrieved_docs': float(avg_retrieved_docs),
            'avg_correctly_matched_docs': float(avg_correctly_matched_docs),
            'avg_target_rank': float(avg_target_rank),
            'target_paragraphs_found_rate': float(target_paragraphs_found_rate),
            'top15_found_rate': float(top15_found_rate),
            'total_target_paragraphs': total_target_paragraphs,
            'target_paragraphs_found': target_paragraphs_found,
            'top15_found': top15_found,
            'similarity_scores': [r['similarity_score'] for r in successful_results],
            'guide_similarity_scores': [r['guide_similarity_score'] for r in successful_results],
            'search_counts': [r['search_count'] for r in successful_results],
            'retrieval_scores': [r['retrieval_score'] for r in successful_results],
            'correct_paragraph_overlaps': correct_paragraph_overlaps
        }
    
    def _save_final_results(self, results: List[Dict], summary: Dict, model_id: str, max_samples: int, topk: int, retriever_info: Dict):
        """Save final results and summary"""
        if not hasattr(self, 'current_run_dir'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_run_dir = os.path.join(self.results_dir, "runs", f"run_{timestamp}")
            os.makedirs(self.current_run_dir, exist_ok=True)
        
        # Save to current run directory with simple names
        results_file = os.path.join(self.current_run_dir, "results.json")
        summary_file = os.path.join(self.current_run_dir, "summary.csv")
        details_file = os.path.join(self.current_run_dir, "details.csv")
        
        # Save detailed results
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'model_id': model_id,
                    'max_samples': max_samples,
                    'topk': topk,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'qa_data_path': self.echr_qa_path,
                    'guide_data_path': self.echr_guide_path,
                    'retriever_info': retriever_info
                },
                'summary': summary,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        # Save summary CSV with retriever info
        summary_with_retriever = summary.copy()
        summary_with_retriever.update({
            'retriever_type': retriever_info.get('retriever_type', 'unknown'),
            'retriever_model': retriever_info.get('model_name', 'unknown'),
            'index_type': retriever_info.get('index_type', 'unknown'),
            'server_responsive': retriever_info.get('server_responsive', False)
        })
        
        summary_df = pd.DataFrame([summary_with_retriever])
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed results CSV
        if results:
            results_df = pd.DataFrame([
                {
                    'index': r['index'],
                    'success': r['success'],
                    'similarity_score': r['similarity_score'],
                    'guide_similarity_score': r['guide_similarity_score'],
                    'search_count': r['search_count'],
                    'retrieval_score': r['retrieval_score'],
                    'duration': r['duration'],
                    'question_length': len(r['question']),
                    'answer_length': len(r['clean_answer']),
                    'ground_truth_guide_length': len(r.get('ground_truth_guide_text', '')),
                    'target_guide': r.get('target_guide', ''),
                    'matched_guides': ','.join(r.get('matched_guides', [])),
                    'paragraph_overlap': r.get('paragraph_overlap', 0),
                    'correct_paragraph_overlap': r.get('correct_paragraph_overlap', 0),
                    'target_paragraphs_count': len(r.get('target_paragraphs', [])) if r.get('target_paragraphs') else 0,
                    'found_paragraphs_count': len(r.get('found_paragraphs', [])),
                    'correctly_matched_docs': len(r.get('correctly_matched_docs', [])),
                    'correctly_matched_paragraphs': ','.join(map(str, r.get('correctly_matched_paragraphs', []))),
                    'total_retrieved_docs': r.get('total_retrieved_docs', 0),
                    'target_paragraph_ranks': str(r.get('target_paragraph_ranks', {})),
                    'avg_target_rank_per_sample': np.mean(list(r.get('target_paragraph_ranks', {}).values())) if r.get('target_paragraph_ranks') else 0.0,
                    'target_paragraphs_found_in_sample': len(r.get('target_paragraph_ranks', {})),
                    'top15_found_in_sample': sum(1 for rank in r.get('target_paragraph_ranks', {}).values() if rank <= 15),
                    'error': r['error']
                }
                for r in results
            ])
            
            results_df.to_csv(details_file, index=False)
        
        # Copy results to 'latest' directory for quick access
        latest_dir = os.path.join(self.results_dir, "latest")
        import shutil
        shutil.copy2(results_file, os.path.join(latest_dir, "results.json"))
        shutil.copy2(summary_file, os.path.join(latest_dir, "summary.csv"))
        if os.path.exists(details_file):
            shutil.copy2(details_file, os.path.join(latest_dir, "details.csv"))
        
        print(f"\n=== Evaluation Complete ===")
        print(f"📁 Run Directory: {self.current_run_dir}")
        print(f"📄 Results: {os.path.basename(results_file)}")
        print(f"📊 Summary: {os.path.basename(summary_file)}")
        if os.path.exists(details_file):
            print(f"📋 Details: {os.path.basename(details_file)}")
        print(f"🔗 Latest results also available in: {os.path.join(self.results_dir, 'latest')}")
        
        print(f"\n📂 Directory Structure:")
        print(f"  {self.current_run_dir}/")
        print(f"  ├── results.json     # Complete results with metadata")
        print(f"  ├── summary.csv      # Summary statistics")
        print(f"  ├── details.csv      # Detailed per-sample results")
        print(f"  └── intermediate/    # Step-by-step results")
        
        print(f"\nRetriever Information:")
        print(f"Type: {retriever_info.get('retriever_type', 'unknown')}")
        print(f"Model: {retriever_info.get('model_name', 'unknown')}")
        print(f"Index Type: {retriever_info.get('index_type', 'unknown')}")
        print(f"Server Responsive: {retriever_info.get('server_responsive', False)}")
        print(f"\nSummary Statistics:")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Similarity: {summary['avg_similarity']:.3f}")
        print(f"Average Guide Similarity: {summary['avg_guide_similarity']:.3f}")
        print(f"Average Search Count: {summary['avg_search_count']:.1f}")
        print(f"Average Retrieval Score: {summary['avg_retrieval_score']:.3f}")
        print(f"Guide Found Rate: {summary['guide_found_rate']:.1%}")
        print(f"Correct Match Overlap: {summary['avg_correct_paragraph_overlap']:.1f}")
        print(f"Correct Match Rate: {summary['correct_match_rate']:.1%}")
        print(f"Average Retrieved Docs: {summary['avg_retrieved_docs']:.1f}")
        print(f"Average Correctly Matched Docs: {summary['avg_correctly_matched_docs']:.1f}")
        print(f"Average Target Paragraph Rank: {summary['avg_target_rank']:.1f}")
        print(f"Target Paragraphs Found Rate: {summary['target_paragraphs_found_rate']:.1%}")
        print(f"Top-15 Found Rate: {summary['top15_found_rate']:.1%}")
        print(f"Target Paragraphs Found: {summary['target_paragraphs_found']}/{summary['total_target_paragraphs']}")
        print(f"Top-15 Found: {summary['top15_found']}/{summary['total_target_paragraphs']}")
        print(f"Average Duration: {summary['avg_duration']:.1f}s")


def main():
    """Main function to run the evaluation"""
    
    # Configuration
    # model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3"
    model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3"  # Alternative smaller model
    
    max_samples = 3  # Start with a small number for testing
    topk = 15
    
    # Initialize evaluator
    evaluator = ECHRTestEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation(
        model_id=model_id,
        max_samples=max_samples,
        topk=topk
    )
    
    return results


if __name__ == "__main__":
    main()
