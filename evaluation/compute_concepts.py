import spacy
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.linking import EntityLinker
from datasets import load_dataset
import os
import numpy as np


print("Loading spaCy model...")
nlp = spacy.load("en_core_sci_lg")

if "scispacy_linker" not in nlp.pipe_names:
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, 
        "linker_name": "umls",
        "threshold": 0.7
    })

print("Model loaded. Processing dataset...")
dataset_path = 'test.jsonl'
dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset
references = [ex["target"] for ex in dataset]
print("Loaded dataset and references")

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

print("Defined functions")

generated_summaries_path = "generated_summaries.txt"
with open(generated_summaries_path, "r") as f:
    generated_summaries = f.read().splitlines()
    print("Loaded generated summaries")
    print(generated_summaries[:3])


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
    recall    = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    f1_scores.append(f1)

    ref_types = extract_semtypes(reference)
    gen_types = extract_semtypes(gen_summary)

    coverage_rate = len(gen_types & ref_types) / (len(ref_types) + 1e-6)

    coverage_scores.append(coverage_rate)

    print("Calculated scores for text:", gen_summary)


print("done\n")


cui_f1_stats = stats(f1_scores, scale=100)
coverage_rate_stats = stats(coverage_scores, scale=100)

print("CUI F1:", cui_f1_stats)
print("Coverage rate:", coverage_rate_stats)