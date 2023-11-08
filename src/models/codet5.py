# source code to train CodeT5 model for fine-tuning.
import os
import json
import argparse
import evaluate
import numpy as np
from typing import *
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset, load_metric
from src.evaluator.CodeBLEU.calc_code_bleu import codebleu_fromstr
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, pipeline

TREE_SITTER_LANG_CODE_CORRECTIONS = {
    "cs": "c_sharp"
}

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
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="training/evaluation mode")
    # parser.add_argument("--checkpoint_path", type=str, default=None)
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
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        # pad_to_multiple_of=64,
        # return_tensors="pt"
    )
    # intialize trainer.
    trainer = Trainer(
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
    bleu_metric = evaluate.load("bleu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
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
    LOG_FILE_PATH = os.path.join(args.output_dir, "train_logs.jsonl")
    # interesting models:
    # 1. Salesforce/codet5-base-multi-sum (base)
    # 2. Salesforce/codet5p-2b (extra large)
    # 3. Salesforce/codet5p-770m (large)
    # 4. Salesforce/codet5-large
    if args.mode == "train": 
        dataset = dataset.map(preprocess_function, batched=True, desc="Running tokenizer")
        train(args)
    elif args.mode == "eval": 
        test_dataset = SimpleDataset(data=[rec[input_key] for rec in dataset["test"]])
        eval(args)