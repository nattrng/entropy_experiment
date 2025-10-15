import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from evals.arc import ARC
from evals.mmlu import MMLU
from evals.hellaswag import HellaSwag

device = 'mps' if torch.mps.is_available() else 'cpu'
dtype = torch.bfloat16


model_list = ['Qwen/Qwen3-Next-80B-A3B-Instruct', 'NousResearch/Hermes-4-70B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'mistralai/Mistral-Small-3.2-24B-Instruct-2506']
models = {label: [[], []] for label in model_list}
evals = [ARC("ARC-Challenge", "test"), MMLU("all", "test"), HellaSwag("default", "validation")]

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


for label in models:

    model = AutoModelForCausalLM.from_pretrained(label, torch_dtype=dtype, device_map=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(label, trust_remote_code=True)
    
    for task in evals:
        if task == evals[0]:
            num_examples = task.num_examples()
        else:
            num_examples = 5000
        for i in tqdm(range(num_examples), desc=f'{label}'):
            ex = task.get_example(i)
            shannon_entropy = lambda probs: -(probs.float().clamp(min=1e-10) * torch.log2(probs.float().clamp(min=1e-10) + 1e-9)).sum()
            
            model_probs, model_pred = get_logprobs(model, tokenizer, ex["messages"][:-1], ex["letters"])
            entropy = shannon_entropy(model_probs)

            if task.evaluate(ex, model_pred) == True: 
                models[label][0].append(entropy) 
            else: models[label][1].append((entropy))


    del model
    del tokenizer
    if device == "mps":
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()

true_entropies = []
false_entropies = []

for label in models:
    true_entropy_mean = torch.tensor(models[label][0]).sum() / torch.tensor(models[label][0]).numel()
    true_entropies.append(true_entropy_mean.item())
    false_entropy_mean = torch.tensor(models[label][1]).sum() / torch.tensor(models[label][1]).numel()
    false_entropies.append(false_entropy_mean.item())
    print(f"{label}: True Average Shannon Entropy: {true_entropy_mean:.3f} | False Average Shannon Entropy {false_entropy_mean:.3f}")

x = np.arange(len(model_list))
width = 0.35

plt.figure(figsize=(12, 6)) 
plt.bar(x - width/2, true_entropies, width, label='Entropy (True)', color = 'blue')
plt.bar(x + width/2, false_entropies, width, label='Entropies (False)', color = 'gray')

plt.title('Collated Entropies Over Several Models')
plt.xlabel('Models')
plt.ylabel('Shannon Entropy')

plt.xticks(x, model_list, rotation=45, ha='right')
plt.legend()
plt.savefig('entropy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# I changed it so that it processes each model sequentially rather than switching back and forth (bloats up memory swap).
