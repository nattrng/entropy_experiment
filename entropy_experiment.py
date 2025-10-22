import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from evals.mcq.arc import ARC
from evals.mcq.mmlu import MMLU
from evals.mcq.hellaswag import HellaSwag
from models import model_list
import os

os.makedirs("model_entropy_charts", exist_ok=True)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
dtype = torch.bfloat16
batch_size = 300

evals = [
    [ARC("ARC-Challenge", "test"), "ARC-Challenge"]#, 
    [MMLU("all", "test"), "MMLU"], 
    [HellaSwag("default", "validation"), "HellaSwag"]
]

def get_logprobs(model, tokenizer, messages, letters):
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = messages[0]['content']
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    
    letter_ids = []
    for l in letters:
        tokens = tokenizer.encode(l, add_special_tokens=False)
        if len(tokens) > 0:
            letter_ids.append(tokens[0])
        else:
            tokens = tokenizer.encode(f" {l}", add_special_tokens=False)
            letter_ids.append(tokens[0] if len(tokens) > 0 else 0)
    
    letter_logits = logits[letter_ids]
    probs = torch.softmax(letter_logits, dim=0).float()
    return probs, letters[probs.argmax().item()]

def save_graph(model_name, eval_names, eval_entropies):

    avg_true_eval_entropy = [torch.tensor(values[0]).mean() for values in eval_entropies.values()]
    avg_false_eval_entropy = [torch.tensor(values[1]).mean() for values in eval_entropies.values()]


    x = np.arange(len(eval_names))
    width = 0.35
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, avg_true_eval_entropy, width, label='Entropy (True)', color = 'blue')
    plt.bar(x + width/2, avg_false_eval_entropy, width, label='Entropies (False)', color = 'gray')

    plt.title(f'{model_name}: Entropy Evaluations')
    plt.xlabel('Benchmarks')
    plt.ylabel('Shannon Entropy')

    plt.xticks(x, eval_names, rotation=45, ha='right')
    plt.legend()
    edited_name = model_name.replace("/","_")
    plt.savefig(os.path.join("model_entropy_charts", f'{edited_name}_entropy_comparisons.png'), dpi=300, bbox_inches='tight')


for label in model_list:
    model = AutoModelForCausalLM.from_pretrained(label, dtype=dtype, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(label, trust_remote_code=True)
    eval_names = [eval[1] for eval in evals]
    eval_entropies = {eval[1]: [[], []] for eval in evals} # idx 1 denotes true while idx 2 denotes false
    
    for eval in evals:
        for batch in tqdm(range(batch_size, eval[0].num_examples(), batch_size), desc=f'{eval[1]}'):
            for i in range(batch - batch_size, batch):
                ex = eval[0].get_example(i)
                shannon_entropy = lambda probs: -(probs.float().clamp(min=1e-10) * torch.log2(probs.float().clamp(min=1e-10) + 1e-9)).sum()
                
                model_probs, model_pred = get_logprobs(model, tokenizer, ex["messages"][:-1], ex["letters"])
                entropy = shannon_entropy(model_probs)
    
                if eval[0].evaluate(ex, model_pred) == True: 
                    eval_entropies[eval[1]][0].append(entropy) 
                else: eval_entropies[eval[1]][1].append(entropy)
    
    save_graph(label, eval_names, eval_entropies)

    del model
    del tokenizer
    if device == "mps":
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()
