# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.

""" Finetuning a 🤗 Transformers model for Signal Detection - Binary Classification."""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
# from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from datasets import DatasetDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from transformers.utils import get_full_repo_name

import csv
import pandas as pd
import numpy as np

from scipy.stats import mode

logger = get_logger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model for Signal Detection")
    
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
  
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
  
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
  
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
  
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
  
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
  
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
  
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--output_dir_ensemble", type=str, default=None, help="Where to store the final model.")
    

    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
  
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
  
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
  
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
  
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
  
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
  
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training or a validation file.")
        
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. Let the accelerator handle device placement.
    # If using tracking, it is also needed to initialize it here and it will by default pick up all supported trackers
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
    logger.info(accelerator.state, main_process_only=False)
  
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Define multiple seed for the ensemble
    seeds = [42, 123, 2024, 777, 999]
    model_paths = [] # store trained model paths

    """
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    """

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently download the dataset.
 
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
       

    # Labels
    def preprocessing(examples):
        sentences = []
        labels = []    
        for i, text in enumerate(examples["text"]):
            
            sentences.append(text)
            if any(['<SIG0>' in _ for _ in eval(examples["causal_text_w_pairs"][i])]):
                labels.append(1)
            else:
                labels.append(0)
        return {"sentences": sentences, "labels": labels}
        
    raw_datasets = raw_datasets.map(preprocessing, batched=True, remove_columns=raw_datasets['train'].column_names)

    num_labels = 2
  
    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
  
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    """
    # We need to initialize a new model for each seed - this part will be defined after
    """

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = examples["sentences"]
        result = tokenizer(
            texts, 
            padding=padding, 
            max_length=args.max_length, 
            truncation=True
        )

        if "labels" in examples: 
            result["labels"] = examples["labels"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done to max length, the default data collator is used - that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision == "fp16" else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # Train an ensemble of models with different seed
    for seed in seeds:
      
          set_seed(seed)

          # Initialize a new model for each seed
          model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes
          )
    
          # Optimizer
      
          # Split weights in two groups, one with weight decay and the other not.
          no_decay = ["bias", "LayerNorm.weight"]
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
          num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
          if args.max_train_steps is None:
              args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
              overrode_max_train_steps = True
    
          lr_scheduler = get_scheduler(
              name=args.lr_scheduler_type,
              optimizer=optimizer,
              num_warmup_steps=args.num_warmup_steps,
              num_training_steps=args.max_train_steps,
          )
    
          # Prepare everything with our `accelerator`.
          model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
              model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
          )
    
          
    
          # We need to recalculate our total training steps as the size of the training dataloader may have changed
          num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
          if overrode_max_train_steps:
              args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
          # Afterwards we recalculate our number of training epochs
          args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
          # Figure out how many steps we should save the Accelerator states
          if hasattr(args.checkpointing_steps, "isdigit"):
              checkpointing_steps = args.checkpointing_steps
              if args.checkpointing_steps.isdigit():
                  checkpointing_steps = int(args.checkpointing_steps)
          else:
              checkpointing_steps = None
        
    
          
          # Train!
          total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
          logger.info("***** Running training *****")
          logger.info(f"  Num examples = {len(train_dataset)}")
          logger.info(f"  Num Epochs = {args.num_train_epochs}")
          logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
          logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
          logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
          logger.info(f"  Total optimization steps = {args.max_train_steps}")
          # Only show the progress bar once on each machine.
          progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
          completed_steps = 0
          starting_epoch = 0
          # Potentially load in the weights and states from a previous save
          if args.resume_from_checkpoint:
              if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                  accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                  accelerator.load_state(args.resume_from_checkpoint)
                  path = os.path.basename(args.resume_from_checkpoint)
              else:
                  # Get the most recent checkpoint
                  dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                  dirs.sort(key=os.path.getctime)
                  path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
              # Extract `epoch_{i}` or `step_{i}`
              training_difference = os.path.splitext(path)[0]
    
              if "epoch" in training_difference:
                  starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                  resume_step = None
              else:
                  resume_step = int(training_difference.replace("step_", ""))
                  starting_epoch = resume_step // len(train_dataloader)
                  resume_step -= starting_epoch * len(train_dataloader)
    
    
            
          results_dir = os.path.join(args.output_dir, "epoch_results")
          os.makedirs(results_dir, exist_ok=True)  # Ensure the directory exists
    
          
          logger.info(f"***** Training model with seed {seed} *****")
    
          for epoch in range(starting_epoch, args.num_train_epochs):
                model.train()
                if args.with_tracking:
                    total_loss = 0
                for step, batch in enumerate(train_dataloader):
                    # We need to skip steps until we reach the resumed step
                    if args.resume_from_checkpoint and epoch == starting_epoch:
                        if resume_step is not None and step < resume_step:
                            completed_steps += 1
                            continue
                    outputs = model(**batch)
                    loss = outputs.loss
                    # keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    loss = loss / args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
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
    
           
                def evaluate_dataloader(dataloader, split_name):
                    """Helper function to evaluate a dataloader and print metrics."""
                    model.eval()
                    samples_seen = 0
                    all_predictions = []
                    all_references = []
                    input_texts = []
        
                    results_csv_file = os.path.join(results_dir, f"{split_name}_results_epoch_{epoch + 1}.csv")
                    if accelerator.is_main_process:
                        # Initialize the CSV file with a header row
                        with open(results_csv_file, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(["Sentence", "True_Label", "Prediction"])
        
                    for step, batch in enumerate(dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)
                        predictions = outputs.logits.argmax(dim=-1)
                        predictions, references = accelerator.gather((predictions, batch["labels"]))
        
                        input_sentences = accelerator.gather(batch["input_ids"])  # Gather tokenized inputs
                        input_texts.extend(tokenizer.batch_decode(input_sentences, skip_special_tokens=True))  # Decode text
        
                        # If we are in a multiprocess environment, the last batch has duplicates
                        if accelerator.num_processes > 1:
                            if step == len(dataloader) - 1:
                                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                                references = references[: len(dataloader.dataset) - samples_seen]
                            else:
                                samples_seen += references.shape[0]
        
                        all_predictions.extend([_.item() for _ in predictions])
                        all_references.extend([_.item() for _ in references])
        
                    # Write the results to the CSV file
                    if accelerator.is_main_process:
                        with open(results_csv_file, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            for text, ref, pred in zip(input_texts, all_references, all_predictions):
                                writer.writerow([text, ref, pred])
        
                    # Calculate accuracy
                    eval_metric = accuracy_score(all_references, all_predictions)
                    logger.info(f"{split_name.capitalize()} Accuracy of epoch {epoch}: {eval_metric}")
        
                    # Calculate precision, recall, F1, and support for each label
                    precision, recall, f1, support = precision_recall_fscore_support(
                        all_references,
                        all_predictions,
                        average=None  # Metrics for each class separately
                    )
        
                    # Print precision, recall, and F1 score table
                    print(f"\n{split_name.capitalize()} Metrics for Epoch {epoch}")
                    print(f"{'Label':<10}{'Support':<10}{'Precision':<10}{'Recall':<10}{'F1 Score':<10}")
                    print("-" * 50)
                    for i, label in enumerate(["0", "1"]):  # Adjust labels as per dataset
                        print(f"{label:<10}{support[i]:<10}{precision[i]:<10.4f}{recall[i]:<10.4f}{f1[i]:<10.4f}")
        
                # Evaluate 
                # evaluate_dataloader(train_dataloader, "train")
                evaluate_dataloader(eval_dataloader, "eval")
        
                if args.push_to_hub and epoch < args.num_train_epochs - 1:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)
                        repo.push_to_hub(
                            commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                        )
        
                if args.checkpointing_steps == "epoch":
                    output_dir = f"epoch_{epoch}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

        model_save_path_seed = os.path.join(args.output_dir_ensemble, f"model_seed_{seed}.pth")
        torch.save(model.state_dict(), model_save_path_seed)
        model_paths.append(model_save_path_seed)
        logger.info(f"Model with seed {seed} saved at {model_save_path_seed}")

    logger.info("All ensemble models trained and saved.")



    def predict_ensemble(models, dataloader):
        all_preds = []
    
        for model_path in models:
            # Load model and set to evaluation mode
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            preds_list = []
            with torch.no_grad():
                for batch in dataloader:
                    outputs = model(**batch)
                    # Get predicted class indices (not logits)
                    preds_list.append(torch.argmax(outputs.logits, dim=1).cpu().numpy())

            # Store predictions for each model
            all_preds.append(np.concatenate(preds_list))

        # Use majority voting to determine final predictions
        # `mode` returns the most common value along axis=0 (i.e., across all models)
        majority_preds, _ = mode(np.array(all_preds), axis=0)

        # Return the final predictions from the majority vote
        return majority_preds.flatten()

    def evaluate_ensemble(models, dataloader, split_name, logger):
        all_references = []
        all_predictions = []

        # Get the true labels and predictions from the ensemble
        for batch in dataloader:
            true_labels = batch['labels'].cpu().numpy()  # Assuming labels are in the 'labels' field
            all_references.extend(true_labels)

            # Get ensemble predictions for the batch
            predictions = predict_ensemble(models, [batch])
            all_predictions.extend(predictions)

        # Calculate Accuracy
        accuracy = accuracy_score(all_references, all_predictions)
        logger.info(f"Ensemble Accuracy: {accuracy}")
        
        # Calculate precision, recall, F1, and support for each label
        precision, recall, f1, support = precision_recall_fscore_support(
            all_references,
            all_predictions,
            average=None  # Metrics for each class separately
        )
        
        print(f"\nEnsemble Metrics")
        print(f"{'Label':<10}{'Support':<10}{'Precision':<10}{'Recall':<10}{'F1 Score':<10}")
        print("-" * 50)
        for i, label in enumerate([0, 1]):  # Adjust labels based on your dataset
            print(f"{label:<10}{support[i]:<10}{precision[i]:<10.4f}{recall[i]:<10.4f}{f1[i]:<10.4f}")

    
    evaluate_ensemble(models, eval_dataloader, "Eval", logger)


    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    # if args.output_dir is not None:
    #     with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
    #         json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)


if __name__ == "__main__":
    main()
