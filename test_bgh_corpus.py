#!/usr/bin/env python3
"""
BGH Corpus Case Retrieval Test Script

This script tests whether the Search-R1 model's retriever correctly returns relevant cases
from the BGH corpus database when fed with questions from q_a_pairs_with_doknr.jsonl.

The script evaluates:
1. Whether the correct case IDs (from doknr) are retrieved
2. The ranking position of correct cases in retrieval results
3. Overall retrieval performance metrics

Note: This version focuses only on case-level matching, not paragraph-level matching,
since the QA data doesn't contain paragraph ground truth information.

Database: data/bgh_rag/extracted_paragraphs_chunks_512_0.0.jsonl
QA Data: data/q_a_pairs_with_doknr.jsonl
"""

import json
import os
import re
import requests
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set
from datetime import datetime
from tqdm import tqdm
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')

# Settings
TOPK = 10
EVALUATION_TOPK = 800  # Use a larger topk for evaluation to find target cases
CORPUS_PATH = "data/bgh_rag/extracted_paragraphs_chunks_512_0.0.jsonl"
QA_PATH = "data/q_a_pairs_with_corpus_coverage.jsonl"

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Will use simple text matching instead.")

from inference_engine import search_r1_inference, get_or_create_model


# BGH-specific prompt template optimized for German Federal Court of Justice jurisprudence
BGH_PROMPT_TEMPLATE = """
Beantworten Sie die folgende deutsche Rechtsfrage durch klare Identifikation relevanter Rechtsprechung und Rechtsgrundsätze aus der Judikatur des Bundesgerichtshofs (BGH).

Dokumentieren Sie bei jeder neuen Information explizit Ihren Denkprozess in <think> … </think> Tags.

Wenn zusätzliche externe Informationen benötigt werden, formulieren Sie Ihre Suchanfrage präzise als:
<search> Ihre präzise Suchanfrage </search>

Die Antwort wird zurückgegeben als:
<information> … </information>
Sie können wiederholt suchen, bis Sie mit Ihrer Faktenbasis zufrieden sind.

Wenn ausreichende Informationen gesammelt wurden, verfassen Sie Ihre finale Antwort in:
<answer> … </answer>

Richtlinien für <answer>:
1. Strukturieren Sie Ihre Antwort als klare, gut begründete deutsche Rechtsanalyse.
2. Identifizieren Sie klar, welche BGH-Rechtsprechung und Rechtsgrundsätze (Leitsätze) auf jeden Aspekt der Frage anwendbar sind.
3. Behandeln Sie relevante rechtliche Teilfragen einschließlich: Zuständigkeit, anwendbares Recht, Rechtsgrundlage, Voraussetzungen, Verfahren und Auslegung.
4. Paraphrasieren Sie die Frage—verwenden Sie nicht direkt den Wortlaut aus der Eingabe.
5. Zitieren Sie relevante BGH-Entscheidungen mit Aktenzeichen und Kernrechtsgrundsätzen (z.B. BGH, Urt. v. [Datum], Az. [Aktenzeichen], Leitsatz: ...).
6. Beziehen Sie sich auf spezifische Paragraphen und Absätze der relevanten Gesetze (z.B. § 1 Abs. 1 Nr. 2 ESchG).
7. Berücksichtigen Sie die Struktur der BGH-Judikatur: Entscheidungsgründe, Leitsätze und dogmatische Einordnung.

Hinweis: Suchen Sie gezielt nach BGH-Entscheidungen mit KORE-Nummern und spezifischen Rechtsgrundsätzen, die in den Entscheidungsgruenden erläutert werden.

Rechtsfrage: {question}"""


class BGHCorpusTestEvaluator:
    """Evaluator for BGH corpus case retrieval testing"""
    
    def __init__(self, 
                 bgh_qa_path: str = "data/q_a_pairs_with_doknr.jsonl",
                 corpus_path: str = "data/bgh_rag/extracted_paragraphs_chunks_512_0.0.jsonl",
                 results_dir: str = "test_results",
                 preload_model: bool = True,
                 model_id: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3",
                 use_quantization: bool = True,
                 quantization_bits: int = 8,
                 evaluation_topk: int = 100):
        
        self.bgh_qa_path = bgh_qa_path
        self.corpus_path = corpus_path
        self.results_dir = results_dir
        self.preload_model = preload_model
        self.model_id = model_id
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.evaluation_topk = evaluation_topk  # 评估时使用的topk，用于查找目标案例排名
        
        # Create organized directory structure
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "latest"), exist_ok=True)
        
        # Load data
        print("Loading BGH QA data...")
        self.qa_data = self._load_qa_data()
        print(f"Loaded {len(self.qa_data)} BGH QA items")
        
        # Load corpus metadata (for validation, not the full content)
        print("Loading BGH corpus metadata...")
        self.corpus_metadata = self._load_corpus_metadata()
        print(f"Loaded metadata for {len(self.corpus_metadata)} corpus documents")
        
        # Initialize similarity model (optional, for answer comparison)
        self.similarity_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print("Loading similarity model...")
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Similarity model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load similarity model: {e}")
        
        # Preload SearchR1 model if requested
        self.search_model = None
        if self.preload_model:
            print("Preloading SearchR1 model...")
            self.search_model = get_or_create_model(
                model_id=self.model_id,
                use_quantization=self.use_quantization,
                quantization_bits=self.quantization_bits
            )
            print("✅ SearchR1 model preloaded successfully")
    
    def _load_qa_data(self) -> List[Dict]:
        """Load BGH QA data"""
        qa_data = []
        with open(self.bgh_qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        qa_data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line: {line[:100]}... Error: {e}")
        return qa_data
    
    def _load_corpus_metadata(self) -> Dict[str, Dict]:
        """Load corpus metadata to understand available case IDs"""
        corpus_metadata = {}
        
        # Sample a few lines to understand structure and collect case IDs
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 1000:  # Sample first 1000 lines for metadata
                        break
                    try:
                        doc = json.loads(line.strip())
                        case_id = self._extract_case_id_from_content(doc.get('contents', ''))
                        if case_id:
                            corpus_metadata[case_id] = {
                                'doc_id': doc.get('id'),
                                'sample_content': doc.get('contents', '')[:200] + '...'
                            }
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Warning: Corpus file not found: {self.corpus_path}")
        
        return corpus_metadata
    
    def _extract_case_id_from_content(self, content: str) -> Optional[str]:
        """Extract case ID from document content"""
        # Look for patterns like "case id: KORE315142010"
        pattern = r'case id:\s*([A-Z0-9]+)'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    
    def _extract_target_ids_from_qa(self, qa_item: Dict) -> List[str]:
        """Extract target case IDs from BGH QA item"""
        # BGH data has direct doknr field as target case ID
        target_case_id = qa_item.get('doknr', '')
        target_case_ids = [target_case_id] if target_case_id else []
        
        return target_case_ids
    
    def _get_retriever_info(self, retrieval_server_url: str = "http://127.0.0.1:8000") -> Dict:
        """Get retriever information from the retrieval server"""
        try:
            response = requests.get(f"{retrieval_server_url}/info", timeout=10)
            if response.status_code == 200:
                info = response.json()
                info['server_responsive'] = True
                info['server_url'] = retrieval_server_url
                return info
        except:
            pass
        
        try:
            # Fallback: try to get basic status
            response = requests.get(f"{retrieval_server_url}/health", timeout=5)
            if response.status_code == 200:
                return {
                    "retriever_type": "unknown",
                    "model_name": "unknown", 
                    "index_type": "unknown",
                    "server_responsive": True,
                    "server_url": retrieval_server_url
                }
        except Exception as e:
            pass
        
        return {
            "retriever_type": "unknown",
            "model_name": "unknown",
            "index_type": "unknown", 
            "server_responsive": False,
            "server_url": retrieval_server_url,
            "error": "Could not connect to retrieval server"
        }
    
    def _check_case_retrieval(self, search_queries: List[str], target_case_ids: List[str],
                             retrieval_server_url: str = "http://127.0.0.1:8000/retrieve") -> Dict:
        """Check if retrieval found relevant cases using actual retrieval API"""
        
        # Case-only metrics
        found_case_ids = set()
        case_rankings = {}  # Maps case_id to its best rank position
        
        total_docs = 0
        all_retrieved_docs = []
        
        try:
            for query in search_queries:
                payload = {
                    "queries": [query],
                    "topk": self.evaluation_topk,
                    "return_scores": True,
                    "instruction": "Gegeben eine deutsche Rechtsfrage zu Entscheidungen des Bundesgerichtshofs (BGH), finden Sie relevante Fallentscheidungen und Rechtsgrundsätze, die die in der Anfrage erwähnten rechtlichen Probleme, Prinzipien und Doktrinen direkt behandeln. Suchen Sie gezielt nach BGH-Urteilen mit entsprechenden KORE-Nummern, Leitsätzen und Entscheidungsgründen."
                }
                
                response = requests.post(retrieval_server_url, json=payload, timeout=30)
                response.raise_for_status()
                results = response.json()['result']
                
                if results and len(results) > 0:
                    retrieval_result = results[0]  # First query result
                    total_docs += len(retrieval_result)
                    
                    for idx, doc_item in enumerate(retrieval_result):
                        content = doc_item['document']['contents']
                        doc_info = {
                            'rank': idx + 1,
                            'score': doc_item.get('score', 0.0),
                            'content_preview': content[:200] + '...',
                            'query': query
                        }
                        all_retrieved_docs.append(doc_info)
                        
                        # Extract case ID from content
                        case_id = self._extract_case_id_from_content(content)
                        if case_id:
                            found_case_ids.add(case_id)
                            # Record the best (lowest) rank for this case
                            if case_id not in case_rankings or idx + 1 < case_rankings[case_id]:
                                case_rankings[case_id] = idx + 1
                                
        except Exception as e:
            print(f"Error during retrieval: {e}")
        
        # Calculate case-only metrics
        target_case_set = set(target_case_ids) if target_case_ids else set()
        found_target_cases = found_case_ids.intersection(target_case_set)
        
        # Target case rankings
        target_case_rankings = {case_id: case_rankings.get(case_id, -1) 
                              for case_id in target_case_ids if target_case_ids}
        
        return {
            # Case metrics
            'found_case_ids': list(found_case_ids),
            'target_case_ids': target_case_ids,
            'found_target_cases': list(found_target_cases),
            'target_case_rankings': target_case_rankings,
            'case_retrieval_score': len(found_target_cases) / len(target_case_set) if target_case_set else 0.0,
            'total_unique_cases_found': len(found_case_ids),
            
            # Common metrics
            'total_retrieved_docs': total_docs,
            'all_retrieved_docs': all_retrieved_docs[:10],  # Keep only top 10 for logging
        }
    
    def _compute_answer_similarity(self, generated_answer: str, expected_answer: str) -> float:
        """Compute semantic similarity between generated and expected answers"""
        if not generated_answer.strip() or not expected_answer.strip():
            return 0.0
            
        if self.similarity_model:
            try:
                from sentence_transformers import util
                embeddings = self.similarity_model.encode([generated_answer, expected_answer])
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                return max(0.0, similarity)  # Ensure non-negative
            except Exception as e:
                print(f"Warning: Similarity computation failed: {e}")
        
        # Fallback to simple word overlap
        return self._simple_text_similarity(generated_answer, expected_answer)
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
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
        """Evaluate a single BGH QA item"""
        
        print(f"\n=== Evaluating BGH QA Item {index + 1} ===")
        
        # Extract information from BGH QA format
        question = qa_item.get('frage', '').strip()  # 'frage' is the question field in BGH data
        expected_answer = qa_item.get('leitsatz', '')  # 'leitsatz' is the legal principle/answer
        filename = qa_item.get('filename', '')
        
        # Extract target case IDs using BGH-specific logic
        target_case_ids = self._extract_target_ids_from_qa(qa_item)
        
        print(f"Question: {question}")
        print(f"Target case IDs: {target_case_ids}")
        print(f"Expected answer (Leitsatz) length: {len(expected_answer)} chars")
        print(f"Source file: {filename}")
        
        if not question:
            return {
                'success': False,
                'error': 'No question provided',
                'index': index,
                'question': question,
                'target_case_ids': target_case_ids
            }
        
        try:
            # Run Search-R1 inference with BGH-specific prompt template
            start_time = datetime.now()
            generated_answer, log_file, metadata = search_r1_inference(
                question=question,
                model_id=model_id,
                topk=topk,
                prompt_template=BGH_PROMPT_TEMPLATE
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Clean up generated answer
            clean_answer = self._extract_answer_from_response(generated_answer)
            
            # Check case retrieval performance
            search_queries = metadata.get('search_queries', [])
            retrieval_results = self._check_case_retrieval(search_queries, target_case_ids)
            
            # Compute answer similarity
            answer_similarity = self._compute_answer_similarity(clean_answer, expected_answer)
            
            result = {
                'success': True,
                'index': index,
                'question': question,
                'generated_answer': clean_answer,
                'expected_answer': expected_answer,
                'answer_similarity_score': answer_similarity,
                'filename': filename,
                
                # Case results
                'target_case_ids': target_case_ids,
                'found_case_ids': retrieval_results['found_case_ids'],
                'found_target_cases': retrieval_results['found_target_cases'],
                'target_case_rankings': retrieval_results['target_case_rankings'],
                'case_retrieval_score': retrieval_results['case_retrieval_score'],
                'total_unique_cases_found': retrieval_results['total_unique_cases_found'],
                
                # Common fields
                'search_queries': search_queries,
                'search_count': len(search_queries),
                'duration': duration,
                'log_file': log_file,
                'model_metadata': metadata,
                'total_retrieved_docs': retrieval_results['total_retrieved_docs'],
                'sample_retrieved_docs': retrieval_results['all_retrieved_docs']
            }
            
            print(f"✅ Generated answer length: {len(clean_answer)}")
            print(f"📊 Answer similarity: {answer_similarity:.3f}")
            print(f"📋 Target cases found: {len(retrieval_results['found_target_cases'])}/{len(target_case_ids)}")
            print(f"📈 Case retrieval score: {retrieval_results['case_retrieval_score']:.3f}")
            
            print(f"🔍 Search queries: {len(search_queries)}")
            print(f"⏱️  Duration: {duration:.1f}s")
            
            # Print case rankings
            if retrieval_results['target_case_rankings']:
                print("📍 Target case rankings:")
                for case_id, rank in retrieval_results['target_case_rankings'].items():
                    if rank > 0:
                        print(f"   - {case_id}: Rank {rank}")
                    else:
                        print(f"   - {case_id}: Not found")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return {
                'success': False,
                'error': str(e),
                'index': index,
                'question': question,
                'target_case_ids': target_case_ids,
                'filename': filename
            }
    
    def run_evaluation(self, 
                      model_id: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3",
                      max_samples: int = 5,
                      topk: int = 15,
                      retrieval_server_url: str = "http://127.0.0.1:8000") -> Dict:
        """Run full evaluation on BGH QA dataset for case retrieval"""
        
        print(f"=== Starting BGH Corpus Case Retrieval Evaluation ===")
        print(f"Model: {model_id}")
        print(f"Max samples: {max_samples}")
        print(f"TopK for model: {topk}")
        print(f"TopK for evaluation: {self.evaluation_topk}")
        print(f"Database: {self.corpus_path}")
        print(f"QA Data: {self.bgh_qa_path}")
        print(f"Results directory: {self.results_dir}")
        
        # Create run-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.join(self.results_dir, "runs", f"bgh_run_{timestamp}")
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
        
        for i, qa_item in enumerate(tqdm(sample_data, desc="Evaluating BGH QA items")):
            result = self.evaluate_single_qa(qa_item, i, model_id, topk)
            results.append(result)
            
            # Save intermediate results every 5 items
            if (i + 1) % 5 == 0:
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
            return
        
        intermediate_file = os.path.join(self.current_run_dir, "intermediate", f"step_{count:03d}.json")
        
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 Saved intermediate results to {intermediate_file}")
    
    def _compute_case_level_summary(self, successful_results: List[Dict]) -> Dict:
        """Compute case-level summary statistics"""
        if not successful_results:
            return {}
        
        # Basic metrics
        avg_answer_similarity = np.mean([r['answer_similarity_score'] for r in successful_results])
        avg_case_retrieval_score = np.mean([r['case_retrieval_score'] for r in successful_results])
        avg_search_count = np.mean([r['search_count'] for r in successful_results])
        avg_duration = np.mean([r['duration'] for r in successful_results])
        
        # Case retrieval metrics
        total_target_cases = sum([len(r['target_case_ids']) for r in successful_results])
        total_found_target_cases = sum([len(r['found_target_cases']) for r in successful_results])
        total_unique_cases_found = sum([r['total_unique_cases_found'] for r in successful_results])
        
        # Perfect retrieval rate
        perfect_retrievals = [1 if r['case_retrieval_score'] == 1.0 else 0 for r in successful_results]
        perfect_retrieval_rate = np.mean(perfect_retrievals) if perfect_retrievals else 0.0
        
        # Ranking analysis
        all_rankings = []
        for r in successful_results:
            for rank in r['target_case_rankings'].values():
                if rank > 0:
                    all_rankings.append(rank)
        
        avg_rank = np.mean(all_rankings) if all_rankings else 0.0
        median_rank = np.median(all_rankings) if all_rankings else 0.0
        
        # Top-K hit rates
        top_1_hits = sum([1 for rank in all_rankings if rank == 1])
        top_3_hits = sum([1 for rank in all_rankings if rank <= 3])
        top_5_hits = sum([1 for rank in all_rankings if rank <= 5])
        top_10_hits = sum([1 for rank in all_rankings if rank <= 10])
        
        # Calculate precision@K and recall@K
        precision_at_1, precision_at_3, precision_at_5, precision_at_10 = [], [], [], []
        recall_at_1, recall_at_3, recall_at_5, recall_at_10 = [], [], [], []
        
        for r in successful_results:
            target_case_set = set(r['target_case_ids'])
            if not target_case_set:
                continue
            
            target_rankings = r['target_case_rankings']
            
            for k in [1, 3, 5, 10]:
                relevant_in_topk = sum([1 for rank in target_rankings.values() 
                                      if rank > 0 and rank <= k])
                precision_k = relevant_in_topk / k
                recall_k = relevant_in_topk / len(target_case_set)
                
                if k == 1:
                    precision_at_1.append(precision_k)
                    recall_at_1.append(recall_k)
                elif k == 3:
                    precision_at_3.append(precision_k)
                    recall_at_3.append(recall_k)
                elif k == 5:
                    precision_at_5.append(precision_k)
                    recall_at_5.append(recall_k)
                elif k == 10:
                    precision_at_10.append(precision_k)
                    recall_at_10.append(recall_k)
        
        return {
            'avg_answer_similarity': avg_answer_similarity,
            'avg_case_retrieval_score': avg_case_retrieval_score,
            'perfect_retrieval_rate': perfect_retrieval_rate,
            'total_target_cases': total_target_cases,
            'total_found_target_cases': total_found_target_cases,
            'case_recall': total_found_target_cases / total_target_cases if total_target_cases > 0 else 0.0,
            'avg_rank': avg_rank,
            'median_rank': median_rank,
            'total_ranked_cases': len(all_rankings),
            'top_1_hit_rate': top_1_hits / len(all_rankings) if all_rankings else 0.0,
            'top_3_hit_rate': top_3_hits / len(all_rankings) if all_rankings else 0.0,
            'top_5_hit_rate': top_5_hits / len(all_rankings) if all_rankings else 0.0,
            'top_10_hit_rate': top_10_hits / len(all_rankings) if all_rankings else 0.0,
            'precision_at_1': np.mean(precision_at_1) if precision_at_1 else 0.0,
            'precision_at_3': np.mean(precision_at_3) if precision_at_3 else 0.0,
            'precision_at_5': np.mean(precision_at_5) if precision_at_5 else 0.0,
            'precision_at_10': np.mean(precision_at_10) if precision_at_10 else 0.0,
            'recall_at_1': np.mean(recall_at_1) if recall_at_1 else 0.0,
            'recall_at_3': np.mean(recall_at_3) if recall_at_3 else 0.0,
            'recall_at_5': np.mean(recall_at_5) if recall_at_5 else 0.0,
            'recall_at_10': np.mean(recall_at_10) if recall_at_10 else 0.0,
            'avg_search_count': avg_search_count,
            'avg_duration': avg_duration,
            'total_unique_cases_found': total_unique_cases_found
        }
    
    
    def _compute_summary(self, results: List[Dict]) -> Dict:
        """Compute combined summary statistics"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                'total_samples': len(results),
                'successful_samples': 0,
                'error_rate': 1.0,
                'errors': [r.get('error', 'Unknown error') for r in results if not r['success']]
            }
        
        case_summary = self._compute_case_level_summary(successful_results)
        
        summary = {
            'total_samples': len(results),
            'successful_samples': len(successful_results),
            'error_rate': 1.0 - len(successful_results) / len(results),
            'case_level': case_summary
        }
        
        # Add individual metrics for backward compatibility
        summary.update(case_summary)
        return summary
    
    def _save_final_results(self, results: List[Dict], summary: Dict, model_id: str, 
                           max_samples: int, topk: int, retriever_info: Dict):
        """Save comprehensive final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        final_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_id': model_id,
                'max_samples': max_samples,
                'topk': topk,
                'evaluation_topk': self.evaluation_topk,
                'corpus_path': self.corpus_path,
                'qa_path': self.bgh_qa_path,
                'quantization': self.use_quantization,
                'quantization_bits': self.quantization_bits if self.use_quantization else None,
                'dataset_type': 'BGH'
            },
            'retriever_info': retriever_info,
            'summary': summary,
            'detailed_results': results
        }
        
        # Save JSON results
        results_file = os.path.join(self.current_run_dir, f"bgh_corpus_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save CSV summary
        successful_results = [r for r in results if r['success']]
        if successful_results:
            summary_data = []
            for r in successful_results:
                row = {
                    'index': r['index'],
                    'question_preview': r['question'][:100] + '...' if len(r['question']) > 100 else r['question'],
                    'filename': r.get('filename', ''),
                    'answer_similarity': r['answer_similarity_score'],
                    'case_retrieval_score': r['case_retrieval_score'],
                    'target_cases_count': len(r['target_case_ids']),
                    'found_target_cases_count': len(r['found_target_cases']),
                    'total_unique_cases_found': r['total_unique_cases_found'],
                    'search_count': r['search_count'],
                    'duration': r['duration'],
                    'target_case_ids': ','.join(r['target_case_ids']),
                    'found_target_cases': ','.join(r['found_target_cases'])
                }
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = os.path.join(self.current_run_dir, f"bgh_corpus_summary_{timestamp}.csv")
            summary_df.to_csv(summary_csv, index=False)
        
        # Copy to latest directory
        latest_dir = os.path.join(self.results_dir, "latest")
        latest_results = os.path.join(latest_dir, "bgh_corpus_results_latest.json")
        with open(latest_results, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Print final summary
        print(f"\n🎯 === Final BGH Evaluation Summary ===")
        print(f"📊 Samples: {summary['successful_samples']}/{summary['total_samples']}")
        
        if 'case_level' in summary and summary['case_level']:
            case_summary = summary['case_level']
            print(f"\n📋 === Case-Level Retrieval Results ===")
            print(f"📋 Case Retrieval Score: {case_summary['avg_case_retrieval_score']:.3f}")
            print(f"🎯 Perfect Retrieval Rate: {case_summary['perfect_retrieval_rate']:.3f}")
            print(f"📈 Case Recall: {case_summary['case_recall']:.3f}")
            print(f"📍 Average Rank: {case_summary['avg_rank']:.1f}")
            print(f"🥇 Top-1 Hit Rate: {case_summary['top_1_hit_rate']:.3f}")
            print(f"🥈 Top-3 Hit Rate: {case_summary['top_3_hit_rate']:.3f}")
            print(f"🥉 Top-5 Hit Rate: {case_summary['top_5_hit_rate']:.3f}")
            print(f"🏆 Top-10 Hit Rate: {case_summary['top_10_hit_rate']:.3f}")
            print(f"🎯 Precision@1: {case_summary['precision_at_1']:.3f}")
            print(f"🎯 Precision@3: {case_summary['precision_at_3']:.3f}")
            print(f"🎯 Precision@5: {case_summary['precision_at_5']:.3f}")
            print(f"🎯 Precision@10: {case_summary['precision_at_10']:.3f}")
            print(f"📈 Recall@1: {case_summary['recall_at_1']:.3f}")
            print(f"📈 Recall@3: {case_summary['recall_at_3']:.3f}")
            print(f"📈 Recall@5: {case_summary['recall_at_5']:.3f}")
            print(f"📈 Recall@10: {case_summary['recall_at_10']:.3f}")
        
        
        print(f"\n📝 === Answer Quality ===")
        if 'avg_answer_similarity' in summary:
            print(f"📝 Answer Similarity: {summary['avg_answer_similarity']:.3f}")
        if 'avg_duration' in summary:
            print(f"⏱️  Average Duration: {summary['avg_duration']:.1f}s")
        
        print(f"\n📁 Results saved to:")
        print(f"   - JSON: {results_file}")
        print(f"   - Summary CSV: {summary_csv}")
        print(f"   - Latest: {latest_results}")


def main():
    """Main function to run the BGH corpus case retrieval evaluation"""
    
    parser = argparse.ArgumentParser(description="Run BGH Corpus Case Retrieval Evaluation")
    parser.add_argument(
        "--sample-mode",
        type=str,
        choices=["mini", "medium", "full"],
        default="mini",
        help="Sample mode: mini (10), medium (100), or full (all)."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["3b", "14b", "32b"],
        default="32b",
        help="Model size to use: 3b, 14b, or 32b."
    )
    args = parser.parse_args()

    # Model ID configuration
    model_map = {
        "3b": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3",
        "14b": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3",
        "32b": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3"
    }
    model_id = model_map[args.model_size]
    
    # Initialize evaluator
    evaluator = BGHCorpusTestEvaluator(
        model_id=model_id,
        bgh_qa_path=QA_PATH,
        corpus_path=CORPUS_PATH,
        evaluation_topk=EVALUATION_TOPK
    )
    
    # Sample number configuration
    if args.sample_mode == "mini":
        max_samples = 10
    elif args.sample_mode == "medium":
        max_samples = 100
    else: # full
        max_samples = len(evaluator.qa_data)

    # Run evaluation
    results = evaluator.run_evaluation(
        model_id=model_id,
        max_samples=max_samples,
        topk=TOPK
    )
    
    return results


if __name__ == "__main__":
    main()