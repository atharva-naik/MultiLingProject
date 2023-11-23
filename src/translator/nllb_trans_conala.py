# Use a pipeline as a high-level helper
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

model = "facebook/nllb-200-distilled-600M"
tgt_lang = "deu_Latn"
tgt_column = "german_translation"

dataset = load_dataset("neulab/conala", "mined")
dataset['train'] = dataset['train'].select(range(100000))
pipe = pipeline(f"translation", model=model, device=0, batch_size=256)
res = []
translations = []
for out in tqdm(pipe(KeyDataset(dataset['train'], "intent"), src_lang='eng_Latn', tgt_lang=tgt_lang)):
    for trans in out:
        translations.append(trans['translation_text'])
    
dataset['train'] = dataset['train'].add_column(tgt_column, translations)

dataset['train'].to_json(f"./translator/res/conala_with_{tgt_lang}.jsonl", orient='records', lines=True)

