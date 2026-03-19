from googletrans import Translator

input_file = ""
output_file = "ALL_medical_term_file_English.txt"

translator = Translator()

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        term = line.strip()
        if not term:
            fout.write("\n")
            continue
        try:
            translation = translator.translate(term, src='zh-cn', dest='en')
            fout.write(translation.text + "\n")
            print(f"Translated '{term}' to '{translation.text}'")
        except Exception as e:
            print(f"Error translating '{term}': {e}")
            fout.write(term + "\n")  # fallback: write original if error