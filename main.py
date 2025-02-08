from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding

ds = load_dataset("Silly-Machine/TuPyE-Dataset", "multilabel")

#print("colunas do dataset:", ds["train"].column_names)

ds_split = ds['train'].train_test_split(test_size=0.2)

model_name = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.config.hidden_dropout_prob = 0.3
model.config.attention_probs_dropout_prob = 0.3
model.config.problem_type = "multi_label_classification"

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


tokens_ds = ds_split.map(tokenize_function, batched=True)

def combine_labels(examples):
  label_keys= ['aggressive', 'hate', 'misogyny']
  labels = [examples[key] for key in label_keys]

  return {"labels": labels}
tokens_ds = tokens_ds.map(combine_labels, batched=False)

columns_to_remove = [col for col in tokens_ds['train'].column_names if col not in ['input_ids', 'attention_mask', 'labels']]
tokens_ds = tokens_ds.remove_columns(columns_to_remove)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
      #print("dados recebidos:", inputs)
      labels = inputs.pop("labels")

      if labels is None:
          raise KeyError("Labels not found in inputs")


      labels = torch.tensor(labels).float().to(model.device)


      outputs = model(**inputs)
      logits = outputs.logits

      loss_fct = BCEWithLogitsLoss()
      loss = loss_fct(logits, labels)

      return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    preds = (preds > 0.5).astype(int)
    labels = p.label_ids

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    label_names=["labels"],
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    remove_unused_columns=False,
    dataloader_num_workers=2,

)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokens_ds['train'],
    eval_dataset=tokens_ds['test'],
    compute_metrics=compute_metrics,
    data_collator=data_collator,


)

#print("Colunas disponíveis no dataset:", tokens_ds["train"].column_names)
#print("Exemplo do dataset:", tokens_ds["train"][0])

trainer.train()

