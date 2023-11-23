import os
import googletrans
from tqdm import tqdm
from googletrans import Translator
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset

dataset = load_dataset("neulab/conala", "mined")
trans = Translator()
eng_inputs = dataset['train']['intent'][:1000]
print(len(eng_inputs))
target_lang = "lv"
write_path = f"./translator/res/conala_env_{target_lang}.txt"
lines = []

if os.path.exists(write_path):
    overwrite = "n"
    if overwrite == "n":
        cont = "y"
        if cont == "n": 
            exit()
        else: 
            lines = open(write_path, "r").readlines()
            print(f"File has {len(lines)} lines")
    else:
        open(write_path, "w")

batch_size = 32  # Number of translations to run concurrently

def translate_batch(batch):
    return [trans.translate(text, src="en", dest=target_lang).text for text in batch]

for i in tqdm(range(0, len(eng_inputs), batch_size)):
    batch = eng_inputs[i:i+batch_size]
    translations = translate_batch(batch)
    
    with open(write_path, "a") as f:
        for trans_res in translations:
            f.write(trans_res + "\n")

# with ThreadPoolExecutor(max_workers=batch_size) as executor:
    # for i in tqdm(range(0, len(eng_inputs), batch_size)):
    #     batch = eng_inputs[i:i+batch_size]
    #     translations = list(executor.map(translate_batch, [batch]))
        
    #     with open(write_path, "a") as f:
    #         for trans_res in translations:
    #             print(trans_res)
    #             for trans_str in trans_res:
    #                 f.write(trans_str + "\n")
