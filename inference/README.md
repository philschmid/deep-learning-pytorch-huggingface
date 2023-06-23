# Inference Examples 

## Text Generation Inference 

Run `HuggingFaceH4/starchat-beta` with TGI locally. 

```bash
model=bigscience/bloom-560m
model=HuggingFaceH4/starchat-beta
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