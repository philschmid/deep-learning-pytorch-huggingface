from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
from datasets import concatenate_datasets
import numpy as np

# experiment config
model_id = "google/flan-t5-xl"

# Dataset
dataset_id = "cnn_dailymail"
dataset_config = "3.0.0"
save_dataset_path = "data"
text_column = "article"
summary_column = "highlights"
prompt_start = "Summarize the following news article:\n"
generation_start = "\nSummary:\n"
prompt_template = f"{prompt_start}{{input}}{generation_start}"

# Load dataset from the hub
dataset = load_dataset(dataset_id, name=dataset_config)
# Load tokenizer of FLAN-t5-base
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

prompt_lenght = len(tokenizer(prompt_template.format(input=""))["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_lenght
print(f"Prompt lenght: {prompt_lenght}")
print(f"Max input lenght: {max_sample_length}")

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x[text_column], truncation=True), batched=True, remove_columns=[text_column, summary_column]
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
max_source_length = min(max_source_length, max_sample_length)
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x[summary_column], truncation=True), batched=True, remove_columns=[text_column, summary_column]
)
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# use 95th percentile as max target length
max_target_length = int(np.percentile(target_lenghts, 95))
print(f"Max target length: {max_target_length}")


def preprocess_function(sample, padding="max_length"):
    # created prompted input
    inputs = [prompt_template.format(input=item) for item in sample[text_column]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample[summary_column], max_length=max_target_length, padding=padding, truncation=True
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features))
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

tokenized_dataset["train"].save_to_disk(os.path.join(save_dataset_path, "train"))
tokenized_dataset["test"].save_to_disk(os.path.join(save_dataset_path, "eval"))
