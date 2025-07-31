import transformers
import torch
import random
from datasets import load_dataset
import requests
import os
from datetime import datetime
import re

# question = "How does the Court determine whether a surveillance measure falls within the scope of article 8 of the ECHR, and what conditions must be met for such interference to be considered ""necessary in a democratic society""?"
question = "How does the Court determine whether a surveillance measure falls within the scope of the convention?"
# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3"
# model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_dir = "inference_logs"
os.makedirs(log_dir, exist_ok=True)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"inference_log_{timestamp}.txt")

def print_and_log(text, file_handle=None):
    """Print to console and log to file"""
    print(text)
    if file_handle:
        file_handle.write(text + "\n")
        file_handle.flush()  # Ensure immediate write to file

question = question.strip()
if question[-1] != '?':
    question += '?'
curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Prepare the message
prompt = f"""Answer the following question.

You must conduct all reasoning inside <think> … </think> each time you receive new information.

If you need outside knowledge, issue a search as
<search> your query </search>
The engine will reply with
<information> … </information>
You may search as many times as necessary.

When you have gathered sufficient facts, draft the substantive reply inside
<answer> … </answer> only.

Guidelines for <answer>  
1. Format: 3-7 crisp bullet points **or** 3-6 lean sentences.  
2. Cover **every** sub-issue raised (e.g. scope/interference or positive obligation, legality, legitimate aim, necessity, proportionality, safeguards, balancing). Do **not** omit anything.  
3. Paraphrase the prompt—avoid re-using its wording.  
4. Where helpful, cite 1-2 illustrative Strasbourg cases in-line but keep them brief.  

Example (two-part question)  

<answer>  
• Criterion A …  
• Criterion B … 
• Criterion C …
• Criterion D …
• Illustrative case: *Example v State* § 00.  
</answer>

Question: {question}\n"""

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    device_map="auto"
)

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 5,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

cnt = 0

if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

# 打开日志文件进行写入
with open(log_file, 'w', encoding='utf-8') as log_f:
    # 写入会话开始信息
    header = f"""=== Search-R1 Inference Session ===
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model ID: {model_id}
Question: {question}
Log File: {log_file}
=====================================

"""
    log_f.write(header)
    
    print_and_log('\n\n################# [Start Reasoning + Searching] ##################\n\n', log_f)
    print_and_log(prompt, log_f)
    
    # Encode the chat-formatted prompt and move it to the correct device
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print_and_log(output_text, log_f)
            
            # 写入会话结束信息
            footer = f"""

=== Session Completed ===
Total search iterations: {cnt}
End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
========================
"""
            log_f.write(footer)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # 记录搜索查询
            search_info = f'[Search #{cnt+1}] Query: "{tmp_query}"'
            print_and_log(f'\n{search_info}', log_f)
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        print_and_log(search_text, log_f)

print(f"\ninference process saved to: {log_file}")