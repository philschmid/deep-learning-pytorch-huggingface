# Getting Started with Deep Learning with PyTorch and Hugging Face

This repository contains instructions/examples/tutorials for getting started with Deep Learning Pytorch and Hugging Face libraries like [transformers](https://huggingface.co/docs/transformers/index), [datasets](https://huggingface.co/docs/datasets/index).

### Training

* [Getting started with Pytorch 2.0 and Hugging Face Transformers](./training/pytorch-2-0-bert-text-classification.ipynb) 
* [Fine-tune FLAN-T5 XL/XXL using DeepSpeed & Hugging Face Transformers](./training/deepseed-flan-t5-summarization.ipynb) 
* [Fine-tune FLAN-T5 for chat & dialogue summarization](./training/flan-t5-samsum-summarization.ipynb) 
* [Getting started with Transformers and TPU using PyTorch](./training/accelerate-tpu-bert-text-classification.ipynb)
* [Fine-tune Falcon 180B with DeepSpeed ZeRO, LoRA & Flash Attention](./training/deepseed-falcon-180b-lora-fa.ipynb)
* [Extended Guide: Instruction-tune Llama 2](./training/instruction-tune-llama-2-int4.ipynb)
* [Quantize open LLMs using optimum and GPTQ](./training/optimize-llama-2-gptq.ipynb)

## Requirements

Before we can start make sure you have met the following requirements

* AWS Account with quota
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed
* AWS IAM user [configured in CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) with permission to create and manage ec2 instances


### Commands 

```bash
echo 'export PATH="${HOME}/.local/bin:$PATH"' >> ${HOME}/.bashrc 
````

```bash
watch -n0.1 nvidia-smi
```
