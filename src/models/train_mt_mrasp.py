import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# mt_mrap related imports
from .mt_mrasp.modeling_mt_mrasp import MT_MRASP
from .mt_mrasp.args_mt_mrasp import mt_mrasp_parse_args
from .mt_mrasp.prepare_mt_mrasp_dataset import get_mt_mrasp_loaders

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")


def main():
    # Parse the arguments
    args = mt_mrasp_parse_args()
    
     # ------------------------------------- Helper Functions -------------------------------------
    
    def get_loader():
        train_loaders, val_loaders = get_mt_mrasp_loaders(args)
        return [
            train_loaders["nl_nl"], train_loaders["pl_nl"], train_loaders["nl_pl"],            
            val_loaders["nl_nl"], val_loaders["pl_nl"], val_loaders["nl_pl"],            
        ]
        
    # Check if next batch for mt exists. If not create new loader
    def try_get_mt_batch(loader_, loader_type):
        try:
            batch_data = next(loader_)
            return batch_data, loader_
        except:
            print(f"\nReloading {loader_type} data\n")
            del loader_
            gc.collect()
            nl_nl_loader, pl_nl_loader, nl_pl_loader, _, _ , _ = get_loader()
            if loader_type == "nl_nl":
                batch_data = next(nl_nl_loader)
                return batch_data, nl_nl_loader
            elif loader_type == "pl_nl":
                batch_data = next(pl_nl_loader)
                return batch_data, pl_nl_loader
            elif loader_type == "nl_pl":
                batch_data = next(nl_pl_loader)
                return batch_data, nl_pl_loader
    

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("mt_mrasp_training", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    loaders = get_loader()
    data_loaders = {
        "nl_nl_tr": loaders[0],
        "pl_nl_tr": loaders[1],
        "nl_pl_tr": loaders[2],
        "nl_nl_va": loaders[3],
        "pl_nl_va": loaders[4],
        "nl_pl_va": loaders[5],
    }
    
    max_loader_size = 0
    for data_type, data_loader in data_loaders.items():
        print(f"{data_type} size: {len(data_loader.dataset)}\t{data_type} dataloader size: {len(data_loader)}")
        if "tr" in data_type and len(data_loader) > max_loader_size:
            max_loader_size = len(data_loader)
            
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = MT_MRASP.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    # model.resize_token_embeddings(len(tokenizer))

    # DataLoaders creation:
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(max_loader_size / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.max_train_steps*args.warmup_ratio,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("translation_no_trainer", experiment_config)

    metric = evaluate.load("bleu")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(data_loaders['nl_nl_tr'].dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            nl_nl_loss = 0
            nl_nl_contrast_loss = 0
            
            nl_pl_loss = 0  
            nl_pl_contrast_loss = 0

            pl_nl_loss = 0
            pl_nl_contrast_loss = 0
            
            total_loss = 0
        for step, (nl_nl_batch, nl_pl_batch, pl_nl_batch) in enumerate(zip(data_loaders["nl_nl_tr"], data_loaders["nl_pl_tr"], data_loaders["pl_nl_tr"])):
            
            # nl_nl training
            print("\n\nnl_nl_batch:")
            for key, val in nl_nl_batch.items():
                print(key, val.shape)
            print(f"\n{nl_nl_batch['contrast_mask']}\n")
            nl_nl_batch = {k: v.to(accelerator.device) for k, v in nl_nl_batch.items()}
            outputs = model(**nl_nl_batch)
            loss1 = outputs.loss
            nl_nl_loss += loss1['loss'].detach().float()
            nl_nl_contrast_loss += loss1['contrast_loss'].detach().float()
            
            # nl_pl training
            nl_pl_batch = {k: v.to(accelerator.device) for k, v in nl_pl_batch.items()}
            outputs = model(**nl_pl_batch)
            loss2 = outputs.loss
            nl_pl_loss += loss2['loss'].detach().float()
            nl_pl_contrast_loss += loss2['contrast_loss'].detach().float()
                        
            # pl_nl training
            pl_nl_batch = {k: v.to(accelerator.device) for k, v in pl_nl_batch.items()}
            outputs = model(**pl_nl_batch)
            loss3 = outputs.loss
            pl_nl_loss += loss3['loss'].detach().float()
            pl_nl_contrast_loss += loss3['contrast_loss'].detach().float()
                        
            loss= (
                loss1['loss'].detach().float() + loss2['loss'].detach().float() + loss3['loss'].detach().float() +
                loss1['contrast_loss'].detach().float() + loss2['contrast_loss'].detach().float() + loss3['contrast_loss'].detach().float()            
            )/3
            total_loss += loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            
            if step % args.gradient_accumulation_steps == 0 or step == len(data_loaders["nl_nl_tr"]) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()

        gen_kwargs = {
            "max_length": args.max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
        }
        samples_seen = 0
        for step, (nl_nl_batch, nl_pl_batch, pl_nl_batch) in enumerate(zip(data_loaders["nl_nl_va"], data_loaders["nl_pl_va"], data_loaders["pl_nl_va"])):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(eval_dataloader) - 1:
                        decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        eval_metric = metric.compute()
        print({"bleu": eval_metric["score"]})

        if args.with_tracking:
            accelerator.log(
                {
                    "bleu": eval_metric["score"],
                    "nl_nl_tr_loss": nl_nl_loss.item() / len(data_loaders["nl_nl_tr"]),
                    "nl_pl_tr_loss": nl_pl_loss.item() / len(data_loaders["nl_pl_tr"]),
                    "pl_nl_tr_loss": pl_nl_loss.item() / len(data_loaders["pl_nl_tr"]),
                    "train_loss": total_loss.item() / (len(data_loaders["nl_nl_tr"]) + len(data_loaders["nl_pl_tr"]) + len(data_loaders["pl_nl_tr"])),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_bleu": eval_metric["score"]}, f)


if __name__ == "__main__":
    main()