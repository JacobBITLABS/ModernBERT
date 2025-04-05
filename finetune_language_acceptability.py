import os
import gc
import argparse
import numpy as np
import pandas as pd
from functools import partial

import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr

# ENVIRONMENTS SETUPS #TODO: Move them to train configs!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT for binary classification tasks")
    
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name for fine-tuning"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Pretrained model checkpoint to use"
    )
    parser.add_argument(
        "--train_subset",
        type=int,
        default=10000,
        help="Number of training examples to use"
    )
    parser.add_argument(
        "--do_cleanup",
        action="store_true",
        help="Whether to clean up memory after training"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=8e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=8e-6,
        help="Weight decay"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam beta1"
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.98,
        help="Adam beta2"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Adam epsilon"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (defaults to aai_ModernBERT_{task}_ft)"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/mnt/storage/jnn/outputs/danish/epbibgramnews6",
        help="Base path for dataset files"
    )
    parser.add_argument(
        "--val_subset",
        type=int,
        default=3000,
        help="Number of validation examples to use"
    )
    parser.add_argument(
        "--n_labels",
        type=int,
        default=2,
        help="Number of labels"
    )
    return parser.parse_args()

def cleanup(things_to_delete: list | None = None):
    if things_to_delete is not None:
        for thing in things_to_delete:
            if thing is not None:
                del thing
    gc.collect()
    torch.cuda.empty_cache()

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.training_history = {"train": [], "eval": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:  # Training logs
                self.training_history["train"].append(logs)
            elif "eval_loss" in logs:  # Evaluation logs
                self.training_history["eval"].append(logs)

def create_hf_dataset(correct_file, corrupt_file):
   
    with open(correct_file, 'r', encoding='utf-8') as f:
        correct_sentences = f.readlines()
    correct_labels = [1] * len(correct_sentences)

    with open(corrupt_file, 'r', encoding='utf-8') as f:
        corrupt_sentences = f.readlines()
    corrupt_labels = [0] * len(corrupt_sentences)

    sentences = correct_sentences + corrupt_sentences
    labels = correct_labels + corrupt_labels

    df = pd.DataFrame({'text': sentences, 'label': labels})
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=42)
    split_dataset = dataset.train_test_split(test_size=0.2)

    train_dataset = split_dataset['train']
    valid_dataset = split_dataset['test']

    # Verify the datasets
    print(train_dataset)
    print(valid_dataset)

    return train_dataset, valid_dataset

def compute_metrics(eval_pred):    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return {"accuracy": accuracy_score(predictions, labels)}


def preprocess_function(examples, hf_tokenizer):
    tokenized = hf_tokenizer(examples['text'], truncation=True, padding='max_length') #, max_length=8192)
    return tokenized

def get_label_maps(raw_datasets, train_ds_name):
    labels = raw_datasets[train_ds_name].features["label"]
    id2label = {idx: name.upper() for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None
    label2id = {name.upper(): idx for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None
    return id2label, label2id

def main():
    args = parse_args()
    # Set output_dir if not provided
    if args.output_dir is None:
        args.output_dir = f"aai_ModernBERT_{args.task}_ft"
    
    # Load the dataset
    hf_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    train_dataset, valid_dataset = create_hf_dataset(
        correct_file=os.path.join(args.base_path, "training_correct.txt"),
        corrupt_file=os.path.join(args.base_path, "training_incorrect.txt")
    )
    
    train_dataset = train_dataset.select(range(args.train_subset))
    valid_dataset = valid_dataset.select(range(args.val_subset))
    id2label, label2id = None, None  # get_label_maps(raw_datasets, train_ds_name)
    
    # 3. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenized_train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )
    tokenized_valid_dataset = valid_dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )
    
    print(f"{tokenized_train_dataset}\n")
    print(f"{tokenized_valid_dataset}\n")
    
    # 4. Define the compute metrics function
    task_compute_metrics = partial(compute_metrics)
    
    # 5. Load the model and data collator
    model_additional_kwargs = {"id2label": id2label, "label2id": label2id} if id2label and label2id else {}
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=args.n_labels, **model_additional_kwargs
    )
    hf_data_collator = DataCollatorWithPadding(tokenizer=hf_tokenizer)
    
    # 6. Define the training arguments and trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        adam_epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        bf16_full_eval=True,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=hf_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=hf_tokenizer,
        data_collator=hf_data_collator,
        compute_metrics=task_compute_metrics,
    )
    
    metrics_callback = MetricsCallback()
    trainer.add_callback(metrics_callback)
    trainer.train()
    
    # 7. Get the training results and hyperparameters
    train_history_df = pd.DataFrame(metrics_callback.training_history["train"])
    train_history_df = train_history_df.add_prefix("train_")
    eval_history_df = pd.DataFrame(metrics_callback.training_history["eval"])
    train_res_df = pd.concat([train_history_df, eval_history_df], axis=1)
    args_df = pd.DataFrame([training_args.to_dict()])
    print("[RESULT]: ", args_df)
    
    # 8. Cleanup (optional)
    if args.do_cleanup:
        cleanup(things_to_delete=[trainer, hf_model, hf_tokenizer, tokenized_train_dataset, tokenized_valid_dataset])

if __name__ == "__main__":
    main()
