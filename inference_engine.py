import transformers
import torch
import requests
import os
from datetime import datetime
import re
from typing import Optional, Tuple, Dict, Any
from transformers import BitsAndBytesConfig


class SearchR1Model:
    """Persistent Search-R1 model for reuse across multiple inferences"""
    
    def __init__(self, 
                 model_id: str,
                 use_quantization: bool = True,
                 quantization_bits: int = 8):
        """Initialize and load the model once"""
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        
        print(f"🔄 Loading model {model_id}...")
        self._load_model()
        print(f"✅ Model loaded successfully on {self.device}")
    
    def _load_model(self):
        """Load tokenizer and model"""
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        
        # Configure quantization if requested
        if self.use_quantization:
            if self.quantization_bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.quantization_bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_threshold=6.0
                )
            else:
                raise ValueError("quantization_bits must be 4 or 8")
            
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
    def inference(self,
                  question: str,
                  log_dir: str = "inference_logs",
                  topk: int = 15,
                  retrieval_server_url: str = "http://127.0.0.1:8000/retrieve",
                  temperature: float = 0.7,
                  max_new_tokens: int = 4096) -> Tuple[str, str, Dict[str, Any]]:
        """
        Run inference using the pre-loaded model
        """
        
        # Setup directories
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"inference_log_{timestamp}.txt")
        
        def print_and_log(text, file_handle=None):
            """Print to console and log to file"""
            print(text)
            if file_handle:
                file_handle.write(text + "\n")
                file_handle.flush()
        
        # Prepare question
        question = question.strip()
        if question[-1] != '?':
            question += '?'
        
        # Model-specific settings
        curr_eos = [151645, 151643]  # for Qwen2.5 series models
        curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
        
        # Prepare the prompt
        prompt = f"""
        Answer the following legal question by clearly identifying the relevant guiding case law.

        Each time you receive new information, explicitly document your thought process within <think> … </think> tags.

        If additional external information is required, clearly formulate your query as:
        <search> your precise query </search>

        The response will be returned as:
        <information> … </information>
        You may search repeatedly until satisfied with your factual basis.

        When sufficient information is gathered, draft your final response inside:
        <answer> … </answer>

        Guidelines for <answer>:
        1. Clearly identify which guiding case law (Strasbourg jurisprudence) applies to each aspect of the question.
        2. Format: Provide either 3-7 concise bullet points or 3-6 clear, succinct sentences.
        3. Address every relevant legal sub-issue, including but not limited to: scope/interference, positive obligations, legality, legitimate aim, necessity, proportionality, safeguards, and balancing.
        4. Paraphrase the question—do not directly reuse wording from the prompt.
        5. Cite 1-2 illustrative Strasbourg cases succinctly in-line to support your points.

        Example Response:

        <answer>
        • Clearly paraphrased criterion A ...
        • Clearly paraphrased criterion B ...
        • Clearly paraphrased criterion C ...
        • Illustrative cases: *Example v State* § 00, *Example 2 v State* § 00.
        </answer>

        Question: {question}
"""

        # Define stopping criterion
        class StopOnSequence(transformers.StoppingCriteria):
            def __init__(self, target_sequences, tokenizer):
                self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
                self.target_lengths = [len(target_id) for target_id in self.target_ids]
                self._tokenizer = tokenizer

            def __call__(self, input_ids, scores, **kwargs):
                targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

                if input_ids.shape[1] < min(self.target_lengths):
                    return False

                for i, target in enumerate(targets):
                    if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                        return True

                return False

        def get_query(text):
            pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[-1].strip()
            else:
                return None

        def search(query: str):
            try:
                payload = {
                    "queries": [query],
                    "topk": topk,
                    "return_scores": True
                }
                response = requests.post(retrieval_server_url, json=payload, timeout=30)
                response.raise_for_status()
                results = response.json()['result']
                            
                def _passages2string(retrieval_result):
                    format_reference = ''
                    for idx, doc_item in enumerate(retrieval_result):
                        content = doc_item['document']['contents']
                        title = content.split("\n")[0] if "\n" in content else "No Title"
                        text = "\n".join(content.split("\n")[1:]) if "\n" in content else content
                        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
                    return format_reference

                return _passages2string(results[0])
            
            except Exception as e:
                print(f"Search error: {e}")
                return f"Search failed: {str(e)}"

        # Initialize stopping criteria
        target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, self.tokenizer)])

        # Apply chat template if available
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

        # Metadata for tracking
        metadata = {
            "model_id": self.model_id,
            "question": question,
            "topk": topk,
            "search_count": 0,
            "start_time": datetime.now(),
            "quantization": self.use_quantization,
            "quantization_bits": self.quantization_bits if self.use_quantization else None
        }
        
        final_answer = ""
        search_queries = []
        
        # Main inference loop with logging
        with open(log_file, 'w', encoding='utf-8') as log_f:
            # Write session header
            header = f"""=== Search-R1 Inference Session ===
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model ID: {self.model_id}
Question: {question}
TopK: {topk}
Quantization: {self.use_quantization} ({self.quantization_bits}-bit)
Log File: {log_file}
=====================================

"""
            log_f.write(header)
            
            print_and_log('\n\n################# [Start Reasoning + Searching] ##################\n\n', log_f)
            print_and_log(prompt, log_f)
            
            # Main generation loop
            while True:
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                attention_mask = torch.ones_like(input_ids)
                
                # Generate text with stopping criteria
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=temperature
                )

                # Check if generation ended with EOS token
                if outputs[0][-1].item() in curr_eos:
                    generated_tokens = outputs[0][input_ids.shape[1]:]
                    output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    print_and_log(output_text, log_f)
                    
                    # Extract final answer
                    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
                    answer_matches = answer_pattern.findall(output_text)
                    if answer_matches:
                        final_answer = answer_matches[-1].strip()
                    
                    # Write session footer
                    footer = f"""

=== Session Completed ===
Total search iterations: {metadata['search_count']}
Final Answer: {final_answer[:100]}...
End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
========================
"""
                    log_f.write(footer)
                    break

                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Get search query
                tmp_query = get_query(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
                if tmp_query:
                    metadata['search_count'] += 1
                    search_queries.append(tmp_query)
                    
                    # Log search query
                    search_info = f'[Search #{metadata["search_count"]}] Query: "{tmp_query}"'
                    print_and_log(f'\n{search_info}', log_f)
                    
                    # Perform search
                    search_results = search(tmp_query)
                else:
                    search_results = ''

                # Update prompt with search results
                search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
                prompt += search_text
                print_and_log(search_text, log_f)

        # Update metadata
        metadata['end_time'] = datetime.now()
        metadata['duration'] = (metadata['end_time'] - metadata['start_time']).total_seconds()
        metadata['search_queries'] = search_queries
        metadata['final_answer'] = final_answer
        
        print(f"Inference process saved to: {log_file}")
        
        return final_answer, log_file, metadata


# Global model instance for reuse
_global_model = None

def get_or_create_model(model_id: str, use_quantization: bool = True, quantization_bits: int = 8) -> SearchR1Model:
    """Get existing model or create new one if different"""
    global _global_model
    
    if _global_model is None or _global_model.model_id != model_id:
        if _global_model is not None:
            print("🔄 Model configuration changed, reloading...")
            # Clear previous model from memory
            del _global_model.model
            del _global_model.tokenizer
            torch.cuda.empty_cache()
        
        _global_model = SearchR1Model(
            model_id=model_id,
            use_quantization=use_quantization,
            quantization_bits=quantization_bits
        )
    
    return _global_model


def search_r1_inference(
    question: str,
    model_id: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3",
    log_dir: str = "inference_logs",
    topk: int = 15,
    retrieval_server_url: str = "http://127.0.0.1:8000/retrieve",
    temperature: float = 0.7,
    max_new_tokens: int = 4096,
    use_quantization: bool = True,
    quantization_bits: int = 8
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Search-R1 inference function with model reuse
    """
    
    # Get or create the model (reuses existing if same model_id)
    model = get_or_create_model(model_id, use_quantization, quantization_bits)
    
    # Run inference using the persistent model
    return model.inference(
        question=question,
        log_dir=log_dir,
        topk=topk,
        retrieval_server_url=retrieval_server_url,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
