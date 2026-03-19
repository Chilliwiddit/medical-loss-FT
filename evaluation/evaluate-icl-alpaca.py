import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import gc
from tqdm import tqdm
import evaluate
import time
import spacy
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.linking import EntityLinker



hf_token = ""
login(token=hf_token)


#base_model_id = "Chilliwiddit/Llama-3.1-8B-RConceptM"
base_model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
#base_model_id = "meta-llama/Llama-3.1-8B"


adapter_model_id = "Chilliwiddit/CHQSumm-llama3.1-8B-LoRA-pyTorch"
#adapter_model_id = "Chilliwiddit/llama3.1-8B-LoRA-pyTorch"
#adapter_model_id = "Chilliwiddit/llama3.1-8B-LogitLoss"


print("loading base model")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    revision = "70d37f62f4f475a5a42dbbd5b7aad38f420e0960",
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)

print("loading PEFT adapter")
model = PeftModel.from_pretrained(base_model, adapter_model_id)
model.eval()

print("and tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    adapter_model_id, 
    revision = "70d37f62f4f475a5a42dbbd5b7aad38f420e0960"
    )
print("done")


print("Loading spaCy model...")
nlp = spacy.load("en_core_sci_lg")

if "scispacy_linker" not in nlp.pipe_names:
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, 
        "linker_name": "umls",
        "threshold": 0.7
    })

dataset_path = '/kaggle/input/datasets/nivedhithii/meqsum-test/MeQSum_test.jsonl'
dataset = load_dataset("json", data_files=dataset_path, split="train") # nolist

dataset

N_ICL = 0
PREFIX, SUFFIX = "Example Input", "Example Summary"

retriever = None
train_inputs = []
train_targets = []


if N_ICL > 0:
    print("Loading model...")
    embedder = SentenceTransformer("pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb")

    print("Opening train_inputs...")
    with open('/kaggle/input/train-inputs-tokens/train.inputs.tok', 'r', encoding='utf-8') as f:
        train_inputs = [ln.strip() for ln in f]
    print("Opening train_targets...")
    with open('/kaggle/input/train-targets-tokens/train.target.tok', 'r') as f:
        train_targets = [ln.strip() for ln in f]

    print("Encoding train_inputs")
    X = embedder.encode(train_inputs, convert_to_tensor=False)

    print("Getting nearest beightbours")
    retriever = NearestNeighbors(n_neighbors=N_ICL, metric="cosine")
    retriever.fit(X)
    
    print("Done!")

def get_icl_prompt(text):
    if N_ICL == 0:
        return text

    vec = embedder.encode([text], convert_to_tensor=False)
    distances, indices = retriever.kneighbors(vec)

    prompt = ""
    for i, idx in enumerate(indices[0]):
        inp = train_inputs[idx].replace("\n", "")
        tgt = train_targets[idx].replace("\n", "")
        prompt += f"{PREFIX} {i+1}: {inp}\n{SUFFIX} {i+1}: {tgt}\n##\n"

    prompt += text
    return prompt

inputs_with_icl = [get_icl_prompt(t) for t in dataset["article"]]

for i in inputs_with_icl:
    print(i)
    print("--------")

system_prompt = "You are an expert medical professional. Summarize the radiology report findings into an impression with minimal text"

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

generated_summaries = []
latencies = []
tokens_per_second = []

print("Starting Generation...")

for i, example in tqdm(enumerate(inputs_with_icl), total=len(inputs_with_icl)):

    full_prompt = alpaca_prompt.format(
        system_prompt,
        example
    )

    encoded = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    start_time = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    end_time = time.perf_counter()
    latency = end_time - start_time
    latencies.append(latency) 

    input_len = encoded["input_ids"].shape[1]
    new_tokens = out[0][input_len:]

    number_of_tokens = len(new_tokens)
    token_rate = number_of_tokens / latency if latency > 0 else 0
    tokens_per_second.append(token_rate)

    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if text=="":
        text = "."
    
    generated_summaries.append(text)

    print(text)

    print(f"Generated {i+1} of {len(inputs_with_icl)} summaries")

    if i % 50 == 0:
        torch.cuda.empty_cache()


print("Generation Complete.")

output_file = "generated_summaries.txt"
output_directory = os.path.join("/kaggle/working/", output_file)

with open(output_directory, "w") as f:
    for item in generated_summaries:
        f.write(f"{item}\n")

print("File saved to output path")


torch.cuda.empty_cache()

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

references = [ex["summary"] for ex in dataset]


def extract_cuis(text):
    doc = nlp(text)
    cuis = set()
    for ent in doc.ents:
        # Using newer scispacy syntax (.kb_ents)
        if hasattr(ent._, "kb_ents"):
            for concept in ent._.kb_ents: 
                cuis.add(concept[0]) 
    return cuis

def stats(x, scale=1):
    arr = np.array(x, dtype=float) * scale
    return {
        "avg": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2)
    }

if "scispacy_linker" not in nlp.pipe_names:
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, 
        "linker_name": "umls",
        "threshold": 0.7
    })

linker = nlp.get_pipe("scispacy_linker")


def extract_semtypes(text):
    """
    Extract unique UMLS semantic types from text.
    Returns:
        set of semantic type codes (e.g., {'T047', 'T184'})
    """
    doc = nlp(text)
    semtypes = set()

    for ent in doc.ents:
        # CHANGE: Use .kb_ents instead of .umls_ents
        if not hasattr(ent._, "kb_ents") or not ent._.kb_ents:
            continue

        for cui, score in ent._.kb_ents:
            # linker.kb replaced linker.umls in newer versions
            entity = linker.kb.cui_to_entity.get(cui)
            
            if entity is None:
                continue
            
            # Add all semantic types for this entity
            for sem_type in entity.types:
                semtypes.add(sem_type)

    return semtypes

print("Computing Clinical Concept F1 and coverage score...")
f1_scores = []
coverage_scores = []

for i, gen_summary in enumerate(generated_summaries):
    reference = references[i]
    
    gen_cuis = extract_cuis(gen_summary)
    ref_cuis = extract_cuis(reference)

    tp = len(gen_cuis & ref_cuis)
    fp = len(gen_cuis - ref_cuis)
    fn = len(ref_cuis - gen_cuis)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    f1_scores.append(f1)

    ref_types = extract_semtypes(reference)
    gen_types = extract_semtypes(gen_summary)

    coverage_rate = len(gen_types & ref_types) / (len(ref_types) + 1e-6)

    coverage_scores.append(coverage_rate)

    print("Calculated scores for text:", gen_summary)


print("done")

rouge_result = rouge.compute(predictions=generated_summaries, references=references)
bleu_result = bleu.compute(predictions=generated_summaries, references=references)
bertscore_result = bertscore.compute(predictions=generated_summaries,references=references,lang="en",batch_size=4,device="cuda")

bleu_stats = stats([bleu_result["bleu"] * 100])
rouge_l_stats = stats([rouge_result["rougeL"] * 100])
bertscore_stats = stats(bertscore_result["f1"], scale=100)
cui_f1_stats = stats(f1_scores, scale=100)
coverage_rate_stats = stats(coverage_scores, scale=100)
latency_stats = stats(latencies, scale=1)
token_rate_stats = stats(tokens_per_second, scale=1)

print("BLEU:", bleu_stats)
print("ROUGE-L:", rouge_l_stats)
print("BERTScore F1:", bertscore_stats)
print("CUI F1:", cui_f1_stats)
print("Coverage rate:", coverage_rate_stats)
print("Latency:", latency_stats)
print("Tokens per second:", token_rate_stats)