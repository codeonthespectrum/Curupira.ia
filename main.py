from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

ds = load_dataset("ruanchaves/hatebr")
#print(ds)
#print(ds['train'][0])

model_name = 'neuralmind/bert-large-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(
        examples["instagram_comments"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

tokens_ds = ds.map(tokenize_function, batched=True)
tokens_ds = tokens_ds.rename_column("offensive_language", "labels")
tokens_ds = tokens_ds.map(lambda x: {"labels": [int(x["labels"])]}) 

ds_split = tokens_ds['train'].train_test_split(test_size=0.2)

#print(ds_split['train'][0].keys())

#print(tokens_ds['train'][0])

columns_to_remove = [col for col in tokens_ds['train'].column_names if col not in ['input_ids', 'attention_mask', 'labels']]
tokens_ds = tokens_ds.remove_columns(columns_to_remove)

#print(tokens_ds['train'][0])


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_split['train'],
    eval_dataset=ds_split['test'],
)

trainer.train()