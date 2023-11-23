# Use a pipeline as a high-level helper
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

model = "facebook/nllb-200-distilled-600M"

tgt_langs = ["dan_Latn", "deu_Latn", "fra_Latn", "jpn_Jpan", "rus_Cyrl", "spa_Latn"]
tgt_columns = ["danish_translation", "german_translation", "french_translation", "japanese_translation", "russian_translation", "spanish_translation"]


dataset = load_dataset("json", data_files='translator/res/code_search_net_selected_samples.jsonl', split="train")

# dataset['train'] = dataset['train'].select(range(100000))
print(dataset)
pipe = pipeline(f"translation", model=model, device=0, batch_size=256, max_length=512)
for tgt_lang, tgt_column in zip(tgt_langs, tgt_columns):
    res = []
    translations = []
    for out in tqdm(pipe(KeyDataset(dataset, "func_documentation_string"), src_lang='eng_Latn', tgt_lang=tgt_lang)):
        for trans in out:
            translations.append(trans['translation_text'])
        
    tmp_dataset = dataset.add_column(tgt_column, translations)

    tmp_dataset.to_json(f"./translator/res/conala_with_{tgt_lang}.jsonl", orient='records', lines=True)


# Code-search-net dataset creation
# dataset = load_dataset("code_search_net", split="train")
# print(dataset)
# df = dataset.to_pandas()
# print(len(df))

# sample_size = int(0.054 * len(df))

# print("Sample Size: ", sample_size)

# # Determine the minimum size of any stratum
# min_stratum_size = int(sample_size / df['language'].nunique())

# print(min_stratum_size)

# # Perform stratified sampling
# stratified_sample = df.groupby('language').apply(lambda x: x.sample(min_stratum_size, random_state=42))

# # Reset index
# stratified_sample.reset_index(drop=True, inplace=True)

# print(stratified_sample['language'].value_counts())
# jsonl_file_path = 'translator/res/code_search_net_selected_samples.jsonl'

# # Write DataFrame to JSON Lines file
# stratified_sample.to_json(jsonl_file_path, orient='records', lines=True)