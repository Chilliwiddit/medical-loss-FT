import json
import spacy
from tqdm import tqdm

input_jsonl = ""
output_txt = "sample.txt"

nlp = spacy.load("en_core_sci_md")

unique_medical_concepts = set()

with open(input_jsonl, "r", encoding="utf-8") as f_in:
    for line in tqdm(f_in, desc="Extracting concepts"):
        try:
            data = json.loads(line)
            summary_text = data.get("summary", "")
            
            if summary_text:
                print(f"\nProcessing summary: {summary_text[:100]}...")  
                doc = nlp(str(summary_text))
                for ent in doc.ents:
                    concept = ent.text.strip().lower()
                    if concept:
                        unique_medical_concepts.add(concept) 

        except json.JSONDecodeError:
            continue

with open(output_txt, "w", encoding="utf-8") as f_out:
    for concept in sorted(list(unique_medical_concepts)):
        f_out.write(concept + "\n")

print(f"\nDone! Extracted {len(unique_medical_concepts)} unique concepts to {output_txt}")