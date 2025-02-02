from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn as nn

ds = load_dataset("ruanchaves/hatebr")

model_name = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

model.config.problem_type = "binary_classification"

def tokenize_function(examples):
    return tokenizer(
        examples["instagram_comments"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

tokens_ds = ds.map(tokenize_function, batched=True)
tokens_ds = tokens_ds.rename_column("offensive_language", "labels")
tokens_ds = tokens_ds.map(lambda x: {"labels": float(x["labels"])})

#ds_split = tokens_ds['train'].train_test_split(test_size=0.2)

columns_to_remove = [col for col in tokens_ds['train'].column_names if col not in ['input_ids', 'attention_mask', 'labels']]
tokens_ds = tokens_ds.remove_columns(columns_to_remove)

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits.view(-1)

    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, labels.float())

    return (loss, outputs) if return_outputs else loss
  
ds_split = tokens_ds['train'].train_test_split(test_size=0.2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    label_names=["labels"]
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_split['train'],
    eval_dataset=ds_split['test'],
    
)

trainer.train()