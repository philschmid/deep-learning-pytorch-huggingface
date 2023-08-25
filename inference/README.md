# Inference Examples 

## Text Generation Inference 

Run `HuggingFaceH4/starchat-beta` with TGI locally. 

```bash
model=bigscience/bloom-560m
num_shard=1
quantize=bitsandbytes
max_input_length=1562
max_total_tokens=2048
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -ti -p 8080:80 \
  -e MODEL_ID=$model \
  -e QUANTIZE=$quantize \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest
```

send test request 

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"<|system|>\n<|end|>\n<|user|>\nWhat is Deep Learning?<|end|>\n<|assistant|>","parameters":{"temperature":0.2, "top_p": 0.95, "stop" : ["<|end|>"]}}' \
    -H 'Content-Type: application/json'
```


## Text Generation Infernece GPTQ 

### Lama

Run `TheBloke/Dolphin-Llama2-7B-GPTQ` with TGI locally. 

```bash
# Model config
# model=TheBloke/Llama-2-7b-Chat-GPTQ
# model=T#heBloke/Dolphin-Llama2-7B-GPTQ
model=TheBloke/Llama-2-13b-Chat-GPTQ
num_shard=1
quantize=gptq
max_input_length=1562
max_total_tokens=4096 # 4096
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -ti -p 8080:80 \
  -e MODEL_ID=$model \
  -e QUANTIZE=$quantize \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  -e GPTQ_BITS=$gptq_bits \
  -e GPTQ_GROUPSIZE=$gptq_groupsize \
  -v $volume:/data ghcr.io/huggingface/text-generation-inference:0.9.4
```

send test request 

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.\n<</SYS>>\n\nWhat is 10+10? [\/INST]","parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```
