# Speculative Decoding

## MLP Speculator
```bash
sudo docker run --gpus all -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=ibm-fms/llama3-8b-accelerator \
  -e NUM_SHARD=4 \
  -e MAX_INPUT_TOKENS=1562 \
  -e MAX_TOTAL_TOKENS=2048 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:sha-b70ae09
```

send test request 

```bash
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "stream": false,
  "max_tokens": 250
}' \
    -H 'Content-Type: application/json'
```

## Medusa Speculator
```bash
sudo docker run --gpus all -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=text-generation-inference/Mistral-7B-Instruct-v0.2-medusa \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_TOKENS=1562 \
  -e MAX_TOTAL_TOKENS=2048 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:sha-b70ae09
```

send test request 

```bash
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "user",
      "content": "Write a poem for my three year old"
    }
  ],
  "stream": false,
  "max_tokens": 250
}' \
    -H 'Content-Type: application/json'
```


chat_completions{total_time="2.360607542s" validation_time="256.541µs" queue_time="37.931µs" inference_time="2.36031324s" time_per_token="12.166563ms" seed="Some(5272915472497899851)"}