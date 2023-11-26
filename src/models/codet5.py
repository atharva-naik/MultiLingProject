# source code to train CodeT5 model for fine-tuning.
import os
import json
import torch
import random
import argparse
import evaluate
import warnings
import numpy as np
from typing import *
from tqdm import tqdm
from torch.optim import AdamW
from collections import defaultdict
from losses.info_nce import InfoNCE
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset, load_metric
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.evaluator.CodeBLEU.calc_code_bleu import codebleu_fromstr
from transformers.trainer import PREFIX_CHECKPOINT_DIR, ShardedDDPOption, is_torch_tpu_available, OPTIMIZER_NAME, SCHEDULER_NAME, reissue_pt_warnings, is_sagemaker_mp_enabled, SCALER_NAME, TRAINER_STATE_NAME
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, pipeline, get_linear_schedule_with_warmup

TREE_SITTER_LANG_CODE_CORRECTIONS = {
    "cs": "c_sharp"
}

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class CodeSearchNetTrainer(Trainer):
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = []
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics.append(self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    ))
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            # print(f"\x1b[31;m1DEBUG metrics:\x1b[0m {metrics}")
            # metric_to_check = self.args.metric_for_best_model
            # if not metric_to_check.startswith("eval_"):
            #     metric_to_check = f"eval_{metric_to_check}"
            metric_value = sum([m[f"eval_val{i+1}_bleu"] for i,m in enumerate(metrics)])/4

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

def exact_match_accuracy(refs, preds):
    assert len(refs) == len(preds)
    matches, tot = 0, 0
    for ref, pred in zip(refs, preds):
        ref = ref[0].strip("\n")
        pred = pred.strip("\n")
        matches += int(ref == pred)
        tot += 1

    return matches/tot    

def preprocess_function(examples):
    padding = "max_length"
    max_length = 200

    inputs = [ex for ex in examples[input_key]]
    targets = [ex for ex in examples[target_key]]
    model_inputs = tokenizer(inputs, max_length=max_length, padding=padding, truncation=True)
    labels = tokenizer(targets, max_length=max_length, padding=padding, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

class PredictionDataset(Dataset):
    def __init__(self, data: List[dict]):
        self.data = []
        # do tokenization.
        for i in tqdm(range(len(data)), desc="tokenizing data"):
            task_tag = data[i]['stratify']
            self.data.append(task_tag+" "+data[i]['src_text'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class PretrainingDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer, 
                 max_length=200, padding="max_length"):
        self.data = []
        # do tokenization.
        for i in tqdm(range(len(data)), desc="tokenizing data"):
            task_tag = data[i]['stratify']
            model_inputs = tokenizer(task_tag+" "+data[i]['src_text'], max_length=max_length, padding=padding, truncation=True, return_tensors="pt")
            rev_task_tag = data[i]['tgt_lang']+"-"+data[i]['src_lang']
            labels = tokenizer(data[i]['tgt_text'], max_length=max_length, padding=padding, truncation=True, return_tensors="pt")
            for key in model_inputs:
                model_inputs[key] = model_inputs[key][0]
            model_inputs["labels"] = labels['input_ids'][0]
            # print(rev_task_tag, data[i]['tgt_text'])
            contrast = tokenizer(rev_task_tag+" "+data[i]['tgt_text'], max_length=max_length, padding=padding, truncation=True, return_tensors="pt")
            for key in contrast:
                model_inputs[f"label_{key}"] = contrast[key][0]

            self.data.append(model_inputs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# def pretrain_collate_fn(batch):
#     print(batch)
#     keys = batch[0].keys()
#     batched_inp = {key: [] for key in keys}
#     for i in range(len(batch)):
#         for key in keys:
#             batched_inp[key].append(batch[i][key][0])
#     batched_inp = {key: torch.stack(val) for key, val in batched_inp.items()}

#     return batched_inp

def compute_metrics(pred):
    pred_logits = pred.predictions[0] # n_samples x seqlen x vocab_size
    # pred.predictions[1] # embeddings
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
    pred_ids = np.argmax(pred_logits, axis=-1)
    label_str = [[ele] for ele in tokenizer.batch_decode(pred_ids, group_tokens=False)] # convert to list of list format for references.

    # sacrebleu_output = sacrebleu_metric.compute(predictions=pred_str, references=label_str)
    bleu_output = bleu_metric.compute(predictions=pred_str, references=label_str)
    metric_dict = {
        "bleu_output": bleu_output, 
        # "sacrebleu_output": sacrebleu_output,
        "bleu": bleu_output["bleu"],
        # "sacrebleu": sacrebleu_output["score"]
    }
    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(metric_dict)+"\n")

    return metric_dict

def get_cmdline_args():
    parser = argparse.ArgumentParser(description="Fine-tuning CodeT5 for multilingual text and code tasks")

    # Add arguments
    parser.add_argument("--mode", type=str, default="train", help="training/evaluation mode",
                        choices=["train", "eval", "train_mrasp", "predict_mrasp"])
    parser.add_argument("--tpath", type=str, default="/data/tir/projects/tir3/users/arnaik/conala_transforms.jsonl", 
                        help="code transformations data augmentation")
    parser.add_argument("--no_contrast", action="store_true", help="mRASP without contrastive loss")
    parser.add_argument("--mrasp_lambda", default=0.05, type=float, help="loss function weights")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to checkpoint (model state dict & other stuff) to be loaded")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the model checkpoints and predictions will be stored.")
    # parser.add_argument("--wandb", action="store_true", help="use WandB for logging")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5p-770m", help="Name of the pre-trained weights to load")
    parser.add_argument("--group_by_length", type=bool, default=True, help="Whether to group batches of similar lengths together.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per GPU/TPU core for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps to accumulate gradients before performing a backward/update pass.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="The evaluation strategy to use during training.")
    parser.add_argument("--num_train_epochs", type=int, default=32, help="Total number of training epochs.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Whether to use gradient checkpointing to save memory during training.")
    parser.add_argument("--fp16", type=bool, default=True, help="Whether to use 16-bit (mixed) precision training.")
    parser.add_argument("--save_steps", type=int, default=200, help="Number of steps between saving checkpoints.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Number of training steps between evaluations.")
    parser.add_argument("--logging_steps", type=int, default=200, help="Number of steps between logging training metrics.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for training.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduling.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Total number of checkpoints to save.")
    parser.add_argument("--task", type=str, default="code_x_glue_cc_code_to_code_trans", help="task/dataset for training")
    parser.add_argument("--src", type=str, default="java", help="source language")
    parser.add_argument("--tgt", type=str, default="cs", help="target language")
    parser.add_argument("--eval_accumulation_steps", default=None, type=int, help="to prevent GPU OOM errors for CodeSearchNet")
    parser.add_argument("--out_file", default=None, help="overwrite default file name of `test_output.json` with passed name.")

    # Parse arguments
    args = parser.parse_args()

    return args

class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

def eval(args):
    print("evaluating:", args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    pipe = pipeline("translation", model=model, tokenizer=tokenizer, device="cuda:0")
    print(len(test_dataset))
    label_str = []
    pred_str = []
    bleu_metric = evaluate.load("bleu")
    label_str = [[rec[target_key]] for rec in dataset["test"]]
    pbar = tqdm(pipe(
        test_dataset, max_length=300,
        batch_size=args.per_device_train_batch_size,
    ), total=len(test_dataset))
    for pred in pbar:
        pred_str.extend([item['translation_text'] for item in pred])
    assert len(pred_str) == len(label_str)
    
    bleu_score = bleu_metric.compute(predictions=pred_str, references=label_str)
    if args.task == "code_x_glue_cc_code_to_code_trans":
        lang = TREE_SITTER_LANG_CODE_CORRECTIONS.get(target_key, target_key)
        code_bleu_score = codebleu_fromstr(refs=label_str, hyp=pred_str, lang=lang)
        accuracy = exact_match_accuracy(refs=label_str, preds=pred_str)
    elif args.task == "code_x_glue_tt_text_to_text":
        code_bleu_score = 0
        accuracy = 0
    elif args.task == "code_x_glue_ct_code_to_text":
        if args.src == "en": # code generation
            lang = args.tgt
            code_bleu_score = codebleu_fromstr(refs=label_str, hyp=pred_str, lang=lang)
            accuracy = exact_match_accuracy(refs=label_str, preds=pred_str)
        else: # code summarization
            code_bleu_score = 0
            accuracy = 0
    elif args.task == "neulab/conala": # code generation and summarization.
        if args.src == "en": # code generation
            lang = "python"
            code_bleu_score = codebleu_fromstr(refs=label_str, hyp=pred_str, lang=lang)
            accuracy = exact_match_accuracy(refs=label_str, preds=pred_str)
        else: # code summarization
            code_bleu_score = 0
            accuracy = 0

    print(f'BLEU score: {100*bleu_score["bleu"]:.2f}')
    print(f'CodeBLEU score: {100*code_bleu_score:.2f}')
    print(f'Exact Match Accuracy: {100*accuracy:.2f}')
    print(bleu_score)

    out_file = args.out_file if args.out_file is not None else "test_outputs.json"
    model_output_path = os.path.join(args.output_dir, out_file)
    with open(model_output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "bleu": bleu_score,
            "codebleu": code_bleu_score,
            "preds": pred_str,
            "references": label_str,
        }, f, indent=4)

TRANS_NL_DATA = {
    "src/translator/res/conala_with_dan_Latn.jsonl": "danish",
    "src/translator/res/conala_with_deu_Latn.jsonl": "german",
    "src/translator/res/conala_with_fra_Latn.jsonl": "french",
    "src/translator/res/conala_with_jp_translations.jsonl": "jp",
    "src/translator/res/conala_with_rus_Cyrl.jsonl": "russian",
    "src/translator/res/conala_with_spa_Latn.jsonl": "spanish"
}
def split_data(data, val_size: float=0.005):
    stratification_info = [rec['stratify'] for rec in data]
    train_data, val_data = train_test_split(data, test_size=val_size, stratify=stratification_info)

    return train_data, val_data

def eval_mrasp(model, valloader, args, epoch_i):
    model.eval()
    val_bar = tqdm(enumerate(valloader), 
                   total=len(valloader))
    losses = []
    for step, batch in val_bar:
        with torch.no_grad(): 
            batch = {k: v.cuda() for k,v in batch.items() if not k.startswith("label_")}
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss.item())
            val_bar.set_description(f"{epoch_i+1}/{args.num_train_epochs}: bl: {loss.item():.3f}, l: {np.mean(losses):.3f}")

    return np.mean(losses)

TRANS_LANG_MAP = {
    "da": "danish",
    "de": "german",
    "fr": "french",
    "ja": "jp",
    "ru": "russian",
    "spa": "spanish",
    "es": "spanish",
    "en": "en"
}
def predict_mrasp(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name) 
    assert args.checkpoint_path is not None
    ckpt_dict = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt_dict['model_state_dict'])
    
    data = []
    label_str = []
    pred_str = []

    bleu_metric = evaluate.load("bleu")
    pipe = pipeline('translation', model=model, tokenizer=tokenizer, 
                    device="cuda:0", batch_size=args.per_device_train_batch_size)
    
    accuracy = 0
    code_bleu_score = 0
    # do task-wise processing
    lang = {"py": "python"}.get(args.tgt, args.tgt)
    if args.task == "neulab/conala":
        data_ = load_dataset(args.task, "curated")       
        if args.src == "en": # code generation
            src_key = "intent"
            tgt_key = "snippet"
            task = "codegen"
        elif args.tgt == "en": # code summarization
            src_key = "snippet"
            tgt_key = "intent"
            task = "codesum"
        for inst in data_["test"]:
            data.append({
                "task": task,
                "src_lang": args.src,
                "tgt_lang": args.tgt,
                "stratify": f"{args.src}-{args.tgt}",
                "src_text": inst[src_key],
                "tgt_text": inst[tgt_key]
            })
            # print(inst[tgt_key]) # NOTE: DEBUG
            label_str.append(inst[tgt_key])
    if args.task == "neulab/mconala":
        subset = args.tgt if args.src == "py" else args.src
        data_ = load_dataset(args.task, subset)       
        if args.tgt == "py": # code generation
            src_key = "intent"
            tgt_key = "snippet"
            task = "codegen"
        elif args.src == "py": # code summarization
            src_key = "snippet"
            tgt_key = "intent"
            task = "codesum"
        for inst in data_["test"]:
            data.append({
                "task": task,
                "src_lang": args.src,
                "tgt_lang": args.tgt,
                "stratify": f"{args.src}-{args.tgt}",
                "src_text": inst[src_key],
                "tgt_text": inst[tgt_key]
            })
            # print(inst[tgt_key]) # NOTE: DEBUG
            label_str.append(inst[tgt_key])
    elif args.task == "code_x_glue_tt_text_to_text":
        if args.tgt == "en":
            src_key = "source"
            tgt_key = "target"
            subset =  f"{args.src}_{args.tgt}"
        else:
            src_key = "target"
            tgt_key = "source"
            subset =  f"{args.tgt}_{args.src}"
        data_ = load_dataset(args.task, subset)
        for inst in data_["test"]:
            data.append({
                "task": "doctrans",
                "src_lang": TRANS_LANG_MAP[args.src],
                "tgt_lang": TRANS_LANG_MAP[args.tgt],
                "stratify": f"{TRANS_LANG_MAP[args.src]}-{TRANS_LANG_MAP[args.tgt]}",
                "src_text": inst[src_key],
                "tgt_text": inst[tgt_key]
            })
            label_str.append(inst[tgt_key])
    dataset = PredictionDataset(data)
    for rec in tqdm(pipe(dataset, max_length=300)):
        pred_str.append(rec[0]["translation_text"])
    
    # Compute metrics:
    bleu_score = bleu_metric.compute(predictions=pred_str, references=label_str)
    if args.task == "neulab/conala":
        if args.src == "en":
            accuracy = exact_match_accuracy(preds=pred_str, refs=label_str)
            code_bleu_score = codebleu_fromstr(refs=label_str, hyp=pred_str, lang=lang)
    elif args.task == "neulab/mconala":
        if args.tgt == "py":
            accuracy = exact_match_accuracy(preds=pred_str, refs=label_str)
            code_bleu_score = codebleu_fromstr(refs=label_str, hyp=pred_str, lang=lang)
    # elif args.task == "code_x_glue_tt_text_to_text":
    #     pass

    # save predictions to out_file.
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = args.out_file if args.out_file is not None else "test_outputs.json"
    model_output_path = os.path.join(args.output_dir, out_file)
    print(f"\x1b[34;1mBLEU score:\x1b[0m {100*bleu_score['bleu']:.2f}")
    print(f"\x1b[34;1maccuracy:\x1b[0m {100*accuracy:.2f}")
    print(f"\x1b[34;1mCodeBLEU score:\x1b[0m {100*code_bleu_score:.2f}")
    with open(model_output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "bleu": bleu_score,
            "codebleu": code_bleu_score,
            "preds": pred_str,
            "references": label_str,
        }, f, indent=4)

def train_mrasp(args):
    # load CoNaLa-mined dataset (top-100k) for pre-training.
    from src.datautils import read_jsonl
    conala_mined_dataset = load_dataset("neulab/conala", "mined", split="train[:100000]")
    conala_code_transforms = read_jsonl(args.tpath)
    codegen_data = []
    os.makedirs(args.output_dir, exist_ok=True)
    for i in tqdm(range(len(conala_mined_dataset))):
        inst = conala_mined_dataset[i]
        codegen_data.append({
            "task": "codegen",
            "src_lang": "en",
            "tgt_lang": "py",
            "stratify": "en-py",
            "src_text": inst["intent"],
            "tgt_text": inst["snippet"]
        })
        # assert inst['snippet'] == conala_code_transforms[i]['snippet']
        # for transformed_pl in conala_code_transforms[i]['transforms']:
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
    for path, src_lang in TRANS_NL_DATA.items():
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
                "tgt_text": inst["tgt_text"]
            })
        for i in range(len(trans_data)):
            doctrans_data.append({
                "task": "doctrans",
                "src_lang": "en",
                "tgt_lang": src_lang,
                "stratify": f"en-{src_lang}",
                "src_text": trans_data[i]["intent"],
                "tgt_text": trans_data[i][f"{src_lang}_translation"]
            })
            doctrans_data.append({
                "task": "doctrans",
                "src_lang": src_lang,
                "tgt_lang": "en",
                "stratify": f"{src_lang}-en",
                "src_text": trans_data[i][f"{src_lang}_translation"],
                "tgt_text": trans_data[i]["intent"]
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
            "tgt_text": inst["src_text"]
        })
    print(f"{len(codegen_data)} CodeGen instances")
    print(f"{len(codesum_data)} CodeSum instances")
    print(f"{len(doctrans_data)} DocTrans instances")
    
    cg_train, cg_val = split_data(codegen_data)
    cs_train, cs_val = split_data(codesum_data)
    dt_train, dt_val = split_data(doctrans_data)
    train_data = cg_train + cs_train + dt_train
    val_data = cg_val + cs_val + dt_val
    
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = PretrainingDataset(train_data, tokenizer)
    val_dataset = PretrainingDataset(val_data, tokenizer)
    trainloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    
    # create scheduler and optimizer.
    model = T5ForConditionalGeneration.from_pretrained(args.model_name) 
    num_train_steps = args.num_train_epochs*len(trainloader)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-12)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_train_steps)

    info_nce_loss = InfoNCE(temperature=0.001)
    best_eval_loss = 100000
    model.cuda()
    for epoch_i in range(args.num_train_epochs):
        train_bar = tqdm(enumerate(trainloader), 
                         total=len(trainloader))
        train_losses = []
        for step, batch in train_bar:
            # print(batch)
            for key in batch:
                batch[key] = batch[key].cuda()
            model.train()
            optimizer.zero_grad()
            seq2seq_batch = {k: v for k,v in batch.items() if not k.startswith("label_")}
            
            if not args.no_contrast:
                contrast_batch = {k.replace("label_", ""): v for k,v in batch.items() if k.startswith("label_")}
                seq2seq_outputs = model(**seq2seq_batch)
                e1 = seq2seq_outputs.encoder_last_hidden_state.mean(axis=1)
                e2 = model.encoder(**contrast_batch).last_hidden_state.mean(axis=1)
                loss = seq2seq_outputs.loss+args.mrasp_lambda*info_nce_loss(e1, e2)
            else: 
                seq2seq_outputs = model(**seq2seq_batch)
                loss = seq2seq_outputs.loss

            loss.backward()
            if step % args.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
            train_losses.append(loss.item())
            train_bar.set_description(f"{epoch_i+1}/{args.num_train_epochs}: bl: {loss.item():.3f}, l: {np.mean(train_losses):.3f}")
            if (step != 0 and step % args.eval_steps == 0) or (step+1) == len(trainloader):
                eval_loss = eval_mrasp(model, valloader, args, epoch_i)
                print(f"eval_loss: {eval_loss}")
                ckpt_dict = {
                    "step": step,
                    "epoch": epoch_i+1,
                    "eval_loss": eval_loss,
                    "best_eval_loss": best_eval_loss,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict()
                }
                last_model_save_path = os.path.join(args.output_dir, "last_model.pth")
                torch.save(obj=ckpt_dict, f=last_model_save_path)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    print(f"saving model with best_eval_loss: {best_eval_loss}")
                    best_model_save_path = os.path.join(args.output_dir, "best_model.pth")
                    torch.save(obj=ckpt_dict, f=best_model_save_path)

def train(args):
    """code for fine-tuning CodeT5 for seq2seq translation"""
    # sacrebleu_metric = evaluate.load("sacrebleu")
    if os.path.exists(args.output_dir): # reset training log.
        open(LOG_FILE_PATH, "w")
    print("training:", args.model_name, f"on {args.src} to {args.tgt} {args.num_train_epochs} epochs")
    # training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=args.group_by_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16, 
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=args.save_total_limit,
        metric_for_best_model="bleu",
        load_best_model_at_end=True,
        eval_accumulation_steps=args.eval_accumulation_steps,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        # pad_to_multiple_of=64,
        # return_tensors="pt"
    )
    # intialize trainer.
    trainer = args.trainer_class(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics, 
        # compute_metrics_with_lm if args.use_lm else compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )
    # train the model.
    trainer.train()

# main
if __name__ == "__main__":
    args = get_cmdline_args()
    if args.mode == "train_mrasp":
        train_mrasp(args)
    elif args.mode == "predict_mrasp":
        predict_mrasp(args)
    else:
        bleu_metric = evaluate.load("bleu")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        args.trainer_class = Trainer
        if args.task == "code_x_glue_cc_code_to_code_trans":
            input_key = args.src
            target_key = args.tgt
            dataset = load_dataset(args.task)

        elif args.task == "code_x_glue_tt_text_to_text":
            if args.tgt == "en":
                input_key = "source"
                target_key = "target"
                subset =  f"{args.src}_{args.tgt}"
            else:
                input_key = "target"
                target_key = "source"
                subset =  f"{args.tgt}_{args.src}"
            dataset = load_dataset(args.task, subset)

        elif args.task == "neulab/conala":
            if args.src == "en": # code generation.
                input_key = "intent"
                target_key = "snippet"
            elif args.tgt == "en": # code summarization.
                input_key = "snippet"
                target_key = "intent"
            subset = "curated"
            # split train to get validation (10% of the train dataset)
            dataset = load_dataset(args.task, subset, split=["train[:90%]", "train[90%:]", "test"])
            dataset = {
                "train": dataset[0],
                "validation": dataset[1],
                "test": dataset[2]
            }
        
        elif args.task == "code_x_glue_ct_code_to_text":
            if args.src == "en": # code generation.
                input_key = "docstring"
                target_key = "code"
                subset = args.tgt
            elif args.tgt == "en": # code summarization.
                input_key = "code"
                target_key = "docstring"
                subset = args.src
            dataset = load_dataset(args.task, subset, split=["train", "validation[:20%]", 
                                                            "validation[20%:40%]", "validation[40%:60%]", 
                                                            "validation[60%:80%]", "validation[80%:]", "test"])
            dataset = {
                "train": dataset[0],
                "validation": {
                    "val1": dataset[1],
                    "val2": dataset[2],
                    'val3': dataset[3],
                    "val4": dataset[4],
                },
                "test": dataset[5],
            }
            args.trainer_class = CodeSearchNetTrainer
        LOG_FILE_PATH = os.path.join(args.output_dir, "train_logs.jsonl")
        # interesting models:
        # 1. Salesforce/codet5-base-multi-sum (base)
        # 2. Salesforce/codet5p-2b (extra large)
        # 3. Salesforce/codet5p-770m (large)
        # 4. Salesforce/codet5-large
        if args.mode == "train": 
            if args.task == "code_x_glue_ct_code_to_text":
                dataset["test"] = dataset["test"].map(preprocess_function, batched=True, desc="Running tokenizer")
                dataset["train"] = dataset["train"].map(preprocess_function, batched=True, desc="Running tokenizer")
                dataset["validation"]["val1"] = dataset["validation"]["val1"].map(preprocess_function, batched=True, desc="Running tokenizer")
                dataset["validation"]["val2"] = dataset["validation"]["val2"].map(preprocess_function, batched=True, desc="Running tokenizer")
                dataset["validation"]["val3"] = dataset["validation"]["val3"].map(preprocess_function, batched=True, desc="Running tokenizer")
                dataset["validation"]["val4"] = dataset["validation"]["val4"].map(preprocess_function, batched=True, desc="Running tokenizer")
            elif isinstance(dataset, dict):
                for split in dataset:
                    dataset[split] = dataset[split].map(preprocess_function, batched=True, desc="Running tokenizer")
            else: dataset = dataset.map(preprocess_function, batched=True, desc="Running tokenizer")
            train(args)
        elif args.mode == "eval": 
            test_dataset = SimpleDataset(data=[rec[input_key] for rec in dataset["test"]])
            eval(args)