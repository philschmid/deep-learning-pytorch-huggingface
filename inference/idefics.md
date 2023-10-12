# Idefics inference 

```bash
model=HuggingFaceM4/idefics-9b-instruct
num_shard=1
max_input_length=1562
max_total_tokens=2048

sudo docker run --gpus all -ti -p 8080:80 \
  -e MODEL_ID=$model \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  ghcr.io/huggingface/text-generation-inference:1.1.0
```

send test request 

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"User:<fake_token_around_image>![](https://m.media-amazon.com/images/I/51M87ywnihL._AC_SX679_.jpg)<fake_token_around_image>Can i charge my iphone with this cable?<end_of_utterance>\n","parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```



