#!/usr/bin/env python
# coding: utf-8

# # Efficiently train Large Language Models with LoRA and Hugging Face
#
# In this blog, we are going to show you how to apply [Low-Rank Adaptation of Large Language Models (LoRA)](https://arxiv.org/abs/2106.09685) to fine-tune FLAN-T5 XXL (11 billion parameters) on a single GPU. We are going to leverage Hugging Face [Transformers](https://huggingface.co/docs/transformers/index), [Accelerate](https://huggingface.co/docs/accelerate/index), and [PEFT](https://github.com/huggingface/peft).
#
# You will learn how to:
#
# 1. Setup Development Environment
# 2. Load and prepare the dataset
# 3. Fine-Tune T5 with LoRA and DeepSpeed
# 4. Run Inference with LoRA FLAN-T5
# 5. Cost performance comparison
#
# ### Quick intro: PEFT or Parameter Efficient Fine-tunin
#
# [PEFT](https://github.com/huggingface/peft), or Parameter Efficient Fine-tuning, is a new open-source library from Hugging Face to enable efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. PEFT currently includes techniques for:
#
# - LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
# - Prefix Tuning: [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
# - P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
# - Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)
#
# *Note: This tutorial was created and run on a g5.2xlarge AWS EC2 Instance, including 1 NVIDIA A10G.*

# ## 1. Setup Development Environment
#
# In our example, we use the [PyTorch Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-pytorch.html) with already set up CUDA drivers and PyTorch installed. We still have to install the Hugging Face Libraries, including transformers and datasets. Running the following cell will install all the required packages.

# In[ ]:

# ## 2. Load and prepare the dataset
#
# we will use the [samsum](https://huggingface.co/datasets/samsum) dataset, a collection of about 16k messenger-like conversations with summaries. Conversations were created and written down by linguists fluent in English.
#
# ```python
# {
#   "id": "13818513",
#   "summary": "Amanda baked cookies and will bring Jerry some tomorrow.",
#   "dialogue": "Amanda: I baked cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)"
# }
# ```
#
# To load the `samsum` dataset, we use the **`load_dataset()`** method from the 🤗 Datasets library.

# In[ ]:


from datasets import load_dataset

# Load dataset from the hub
dataset = load_dataset("samsum")

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# Train dataset size: 14732
# Test dataset size: 819


# To train our model, we need to convert our inputs (text) to token IDs. This is done by a 🤗 Transformers Tokenizer. If you are not sure what this means, check out **[chapter 6](https://huggingface.co/course/chapter6/1?fw=tf)** of the Hugging Face Course.

# In[ ]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "google/flan-t5-xxl"

# Load tokenizer of FLAN-t5-XL
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Before we can start training, we need to preprocess our data. Abstractive Summarization is a text-generation task. Our model will take a text as input and generate a summary as output. We want to understand how long our input and output will take to batch our data efficiently.

# In[ ]:


from datasets import concatenate_datasets
import numpy as np

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"]
)
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"]
)
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")


# We preprocess our dataset before training and save it to disk. You could run this step on your local machine or a CPU and upload it to the [Hugging Face Hub](https://huggingface.co/docs/hub/datasets-overview).

# In[ ]:


def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# save datasets to disk for later easy loading
tokenized_dataset["train"].save_to_disk("data/train")
tokenized_dataset["test"].save_to_disk("data/eval")


# The last step before we can start training is to download `nltk` assets, which we use for evaluation.

# In[ ]:


import nltk

nltk.download("punkt", quiet=True)


# ## 3. Fine-Tune T5 with LoRA and bnb int-8
#
# In addition to the LoRA technique, we will use [bitsanbytes LLM.int8()](https://huggingface.co/blog/hf-bitsandbytes-integration) to quantize out frozen LLM to int8. This allows us to reduce the needed memory for FLAN-T5 XXL ~4x.
#
# The first step of our training is to load the model. We are going to use [philschmid/flan-t5-xxl-sharded-fp16](https://huggingface.co/philschmid/flan-t5-xxl-sharded-fp16), which is a sharded version of [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl). The sharding will help us to not run off of memory when loading the model.

# In[ ]:


from transformers import AutoModelForSeq2SeqLM

# huggingface hub model id
model_id = "philschmid/flan-t5-xxl-sharded-fp16"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")


# Now, we can prepare our model for the LoRA int-8 training using `peft`.

# In[ ]:


from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# Define LoRA Config
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
)
# prepare int-8 model for training
model = prepare_model_for_int8_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# As you can see, here we are only training 0.16% of the parameters of the model! This huge memory gain will enable us to fine-tune the model without memory issues.
#
# We want to evaluate our model during training. The `Trainer` supports evaluation during training by providing `compute_metrics`.The most commonly used metric to evaluate summarization task is [rogue_score](https://en.wikipedia.org/wiki/ROUGE_(metric)) short for Recall-Oriented Understudy for Gisting Evaluation). This metric does not behave like the standard accuracy: it will compare a generated summary against a set of reference summaries.
#
# We are going to use `evaluate` library to evaluate the `rogue` score.


from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
)


# The last step is to define the hyperparameters (`TrainingArguments`) we want to use for our training.

# In[ ]:


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

output_dir="lora-flan-t5-xxl"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
		auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)


# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# Let's now train our model and run the cells below. Note that for T5, some layers are kept in `float32` for stability purposes.

# In[ ]:


# train model
trainer.train()

# Save our LoRA model & tokenizer results
peft_model_id="results"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(peft_model_id)