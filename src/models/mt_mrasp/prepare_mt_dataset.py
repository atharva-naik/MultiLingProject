
from ..models.codet5 import PretrainingDataset, CONALA_TRANS_NL_DATA, CODESEARCHNET_TRANS_NL_DATA


def get_mt_mrasp_loaders(args):
    from src.datautils import read_jsonl
    filt_k_conala = args.conala_topk
    conala_mined_dataset = load_dataset("neulab/conala", "mined", 
                                        split=f"train[:{filt_k_conala}]")
    conala_code_transforms = read_jsonl(args.tpath)
    codegen_data = []
    os.makedirs(args.output_dir, exist_ok=True)
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
        #     codegen_data.append({
        #         "task": "codegen",
        #         "src_lang": "en",
        #         "tgt_lang": "py",
        #         "stratify": "en-py",
        #         "src_text": inst["intent"],
        #         "tgt_text": transformed_pl[1]
        #     })
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
    print(f"{len(codegen_data)} CodeGen instances")
    print(f"{len(codesum_data)} CodeSum instances")
    print(f"{len(doctrans_data)} DocTrans instances")
    
    cg_train, cg_val = split_data(codegen_data)
    cs_train, cs_val = split_data(codesum_data)
    dt_train, dt_val = split_data(doctrans_data)

    tasks = ["nl_pl", "pl_nl", "nl_nl"]
    task_to_train_data = {"nl_pl": cg_train, "pl_nl": cs_train, "nl_nl": dt_train}
    task_to_val_data = {"nl_pl  ": cg_val, "pl_nl": cs_val, "nl_nl": dt_val}

    task_to_train_dataloader = dict()
    task_to_val_dataloader = dict()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    for task in tasks:
        print(f"train data size: {len(train_data)}")
        print(f"val data size: {len(val_data)}")
        train_dist = defaultdict(lambda: 0)
        val_dist = defaultdict(lambda: 0)
        for rec in train_data: train_dist[rec['stratify']] += 1
        for rec in val_data: val_dist[rec['stratify']] += 1
        train_dist = dict(train_dist)
        val_dist = dict(val_dist)
        for subset in train_dist:
            print('train-'+subset+f": {round(100*train_dist[subset]/len(train_data), 2)}%")
            print('val-'+subset+f": {round(100*val_dist[subset]/len(val_data), 2)}%")

        train_dataset = PretrainingDataset(train_data, tokenizer, eval=False)
        val_dataset = PretrainingDataset(val_data, tokenizer, eval=True)
        trainloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)

        task_to_train_dataloader[task] = trainloader
        task_to_val_dataloader[task] = valloader

    return task_to_train_dataloader, task_to_val_dataloader