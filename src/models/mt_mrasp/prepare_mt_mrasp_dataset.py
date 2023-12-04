
import os
import random
import json
from collections import defaultdict
from datasets import load_dataset
from typing import *
from tqdm import tqdm
import torch
from torch.utils.data import (
    Dataset,
    DataLoader
)

from datasets import Dataset as HFDataset
from transformers import AutoTokenizer
from ..codet5 import (
    CONALA_TRANS_NL_DATA, 
    CODESEARCHNET_TRANS_NL_DATA,
    split_data
)


class MTLMraspDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer, eval: bool,
                 max_length=200, padding="max_length", dataset="nl_nl"):
        self.data = []
        self.eval = eval
        self.max_length = max_length
        self.padding = padding
        self.tokenizer = tokenizer
        self.tokenized_data_path = f"dataset/tokenized/{dataset}_{len(data)}_eval_{self.eval}"
        if self.tokenized_data_path is not None and os.path.exists(self.tokenized_data_path):
            # If tokenized data exists, load it from disk
            print(f"Reading {dataset} from disk")
            data = HFDataset.load_from_disk(self.tokenized_data_path)
            self.data = []
            for data_point in data:
                self.data.append({k : torch.tensor(v) for k, v in data_point.items()})
            print(f"Length {len(self.data)}")
        else:
            # Tokenize the data and save to disk
            self.data = self._tokenize_data(data)
            if self.tokenized_data_path is not None:
                data = HFDataset.from_list(self.data)
                data.save_to_disk(self.tokenized_data_path)


    def _tokenize_data(self, data):
        tokenized_data = []
        for i in tqdm(range(len(data)), desc="Tokenizing data"):
            task_tag = data[i]['stratify']
            model_inputs = self.tokenizer(task_tag + " " + data[i]['src_text'],
                                        max_length=self.max_length, padding=self.padding,
                                        truncation=True, return_tensors="pt")
            labels = self.tokenizer(data[i]['tgt_text'],
                                max_length=self.max_length, padding=self.padding,
                                truncation=True, return_tensors="pt")
            for key in model_inputs:
                model_inputs[key] = model_inputs[key][0]
            model_inputs["labels"] = labels['input_ids'][0]
            if data[i]['src_trans'] is None:
                data[i]['src_trans'] = data[i]['src_text']
                model_inputs["contrast_mask"] = torch.as_tensor(0)
            else:
                model_inputs["contrast_mask"] = torch.as_tensor(1)
            contrast = self.tokenizer(task_tag + " " + data[i]['src_trans'],
                                    max_length=self.max_length, padding=self.padding,
                                    truncation=True, return_tensors="pt")
            for key in contrast:
                model_inputs[f"contrast_{key}"] = contrast[key][0]

            tokenized_data.append(model_inputs)

        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



def get_mt_mrasp_loaders(args):
    from src.datautils import read_jsonl
    # filt_k_conala = args.conala_topk
    # filt_k_conala = 50000
    filt_k_conala = 10000
    print(f"\nfilt_k_conala: {filt_k_conala}\n")
    conala_mined_dataset = load_dataset("neulab/conala", "mined", split=f"train[:{filt_k_conala}]")
    conala_code_transforms = read_jsonl("/data/tir/projects/tir3/users/arnaik/conala_transforms.jsonl")
    codegen_data = []
    for i in tqdm(range(len(conala_mined_dataset))):
        inst = conala_mined_dataset[i]
        transformed_pl = None
        assert inst['snippet'] == conala_code_transforms[i]['snippet']
        if len(conala_code_transforms[i]['transforms']) > 0:
            transformed_pl = random.sample(conala_code_transforms[i]['transforms'], k=1)[0][1]
        codegen_data.append({
            "task": "codegen",
            "src_lang": "en",
            "tgt_lang": "py",
            "stratify": "en-py",
            "src_text": inst["intent"],
            "tgt_text": inst["snippet"],
            "tgt_trans": transformed_pl,
            "src_trans": None,
        })        
      
    mcodegen_data = []
    doctrans_data = []
    for path, src_lang in CONALA_TRANS_NL_DATA.items():
        trans_data = read_jsonl(path)
        nl_to_trans_nl = {i["intent"]: i[f"{src_lang}_translation"] for i in trans_data}
        for i in tqdm(range(len(codegen_data))):
            inst = codegen_data[i]
            mcodegen_data.append({
                "task": "codegen",
                "src_lang": src_lang,
                "tgt_lang": "py",
                "stratify": f"{src_lang}-py",
                "src_text": nl_to_trans_nl[inst["src_text"]],
                "tgt_text": inst["tgt_text"],
                "src_trans": None,
                "tgt_trans": inst['tgt_trans'],
            })
        for i in range(len(trans_data)):
            doctrans_data.append({
                "task": "doctrans",
                "src_lang": "en",
                "tgt_lang": src_lang,
                "stratify": f"en-{src_lang}",
                "src_text": trans_data[i]["intent"],
                "tgt_text": trans_data[i][f"{src_lang}_translation"],
                "src_trans": trans_data[i]['ras_intent'],
                "tgt_trans": None,
            })
            doctrans_data.append({
                "task": "doctrans",
                "src_lang": src_lang,
                "tgt_lang": "en",
                "stratify": f"{src_lang}-en",
                "src_text": trans_data[i][f"{src_lang}_translation"],
                "tgt_text": trans_data[i]["intent"],
                "tgt_trans": trans_data[i]['ras_intent'],
                "src_trans": None,
            })
    codegen_data = codegen_data+mcodegen_data
    codesum_data = []
    for i in tqdm(range(len(codegen_data))):
        codesum_data.append({
            "task": "codesum",
            "src_lang": codegen_data[i]['tgt_lang'],
            "tgt_lang": codegen_data[i]['src_lang'],
            'stratify': f'py-{codegen_data[i]["src_lang"]}',
            "src_text": inst["tgt_text"],
            "tgt_text": inst["src_text"],
            "src_trans": inst["tgt_trans"],
            "tgt_trans": inst["src_trans"],
        })
        
    args.use_codesearchnet_data = True
    if args.use_codesearchnet_data:
        additional_codegen_data = []
        additional_codesum_data = []
        additional_doctrans_data = []
        for path, lang in CODESEARCHNET_TRANS_NL_DATA.items():
            lang_key = lang if lang != "jp" else "japanese" 
            data_ = read_jsonl(path)
            for rec in data_:
                codelang = rec['language'] # rec['language'] if rec['language'] != "python" else "py"
                additional_codegen_data.append({
                    "task": "codegen",
                    "src_lang": "en",
                    "tgt_lang": codelang,
                    "stratify": f"en-{codelang}",
                    "src_text": rec['func_documentation_string'],
                    "tgt_text": rec["func_code_string"],
                    "src_trans": None,
                    "tgt_trans": None,
                })
                additional_codegen_data.append({
                    "task": "codegen",
                    "src_lang": lang,
                    "tgt_lang": codelang,
                    "stratify": f"{lang}-{codelang}",
                    "src_text": rec[f'{lang_key}_translation'],
                    "tgt_text": rec["func_code_string"],
                    "src_trans": None,
                    "tgt_trans": None,
                })
                additional_codesum_data.append({
                    "task": "codesum",
                    "tgt_lang": "en",
                    "src_lang": codelang,
                    "stratify": f"{codelang}-en",
                    "tgt_text": rec['func_documentation_string'],
                    "src_text": rec["func_code_string"],
                    "src_trans": None,
                    "tgt_trans": None,
                })
                additional_codesum_data.append({
                    "task": "codesum",
                    "tgt_lang": lang,
                    "src_lang": codelang,
                    "stratify": f"{codelang}-{lang}",
                    "tgt_text": rec[f'{lang_key}_translation'],
                    "src_text": rec["func_code_string"],
                    "src_trans": None,
                    "tgt_trans": None,
                })
                additional_doctrans_data.append({
                    "task": "doctrans",
                    "src_lang": "en",
                    "tgt_lang": lang,
                    "stratify": f"en-{lang}",
                    "src_text": rec["func_documentation_string"],
                    "tgt_text": rec[f"{lang_key}_translation"],
                    "src_trans": rec["ras_intent"],
                    "tgt_trans": None,
                })
                additional_doctrans_data.append({
                    "task": "doctrans",
                    "tgt_lang": "en",
                    "src_lang": lang,
                    "stratify": f"{lang}-en",
                    "tgt_text": rec["func_documentation_string"],
                    "src_text": rec[f"{lang_key}_translation"],
                    "src_trans": None,
                    "tgt_trans": rec["ras_intent"],
                })
        codegen_data += additional_codegen_data
        codesum_data += additional_codesum_data
        doctrans_data += additional_doctrans_data
    
    while len(codegen_data) > 100000:
        codegen_data, _ = split_data(codegen_data, val_size=0.2)
        codesum_data, _ = split_data(codesum_data, val_size=0.2)
        doctrans_data, _ = split_data(doctrans_data, val_size=0.2)

    # while len(codegen_data) > 50000:
    #     codegen_data, _ = split_data(codegen_data, val_size=0.2)
    #     codesum_data, _ = split_data(codesum_data, val_size=0.2)
    #     doctrans_data, _ = split_data(doctrans_data, val_size=0.2)

    print(f"{len(codegen_data)} CodeGen instances")
    print(f"{len(codesum_data)} CodeSum instances")
    print(f"{len(doctrans_data)} DocTrans instances")
    
    cg_train, cg_val = split_data(codegen_data)
    cs_train, cs_val = split_data(codesum_data)
    dt_train, dt_val = split_data(doctrans_data)

    expected_train_size = min(len(cg_train), len(cs_train))
    tmp_val_size = len(dt_train) - expected_train_size
    tmp_val_ratio = tmp_val_size / len(dt_train)
    dt_train, _ = split_data(dt_train, tmp_val_ratio)


    print(f"{len(cg_train)} CodeGen Train instances")
    print(f"{len(cs_train)} CodeSum Train instances")
    print(f"{len(dt_train)} DocTrans Train instances")

    tasks = ["nl_pl", "pl_nl", "nl_nl"]
    task_to_train_data = {"nl_pl": cg_train, "pl_nl": cs_train, "nl_nl": dt_train}
    task_to_val_data = {"nl_pl": cg_val, "pl_nl": cs_val, "nl_nl": dt_val}

    task_to_train_dataloader = dict()
    task_to_val_dataloader = dict()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    for task in tasks:
        train_data = task_to_train_data[task]
        val_data = task_to_val_data[task]
        print(f"train data size: {len(train_data)}")
        print(f"val data size: {len(val_data)}")
        train_dist = defaultdict(lambda: 0)
        val_dist = defaultdict(lambda: 0)
        for rec in train_data: train_dist[rec['stratify']] += 1
        for rec in val_data: val_dist[rec['stratify']] += 1
        train_dist = dict(train_dist)
        val_dist = dict(val_dist)
        print("\n"),
        for subset in train_dist:
            print('train-'+subset+f": {round(100*train_dist[subset]/len(train_data), 2)}%")
            print('val-'+subset+f": {round(100*val_dist[subset]/len(val_data), 2)}%")
        print("\n")
        train_dataset = MTLMraspDataset(train_data, tokenizer, eval=False, dataset=task)
        val_dataset = MTLMraspDataset(val_data, tokenizer, eval=True, dataset=task)
        trainloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=args.per_device_train_batch_size, shuffle=False)

        task_to_train_dataloader[task] = trainloader
        task_to_val_dataloader[task] = valloader

    return task_to_train_dataloader, task_to_val_dataloader