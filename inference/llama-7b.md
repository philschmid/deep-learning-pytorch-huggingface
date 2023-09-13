# Llama 2 inference 

```bash
model=meta-llama/Llama-2-7b-chat-hf
token=hf_xxx # replace with your token, which access to the repo
num_shard=1
max_input_length=1562
max_total_tokens=2048

docker run --gpus all -ti -p 8080:80 \
  -e MODEL_ID=$model \
 -e HUGGING_FACE_HUB_TOKEN=$token \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  ghcr.io/huggingface/text-generation-inference:latest
```

send test request 

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\nWhat is 10+10? [\/INST]","parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```