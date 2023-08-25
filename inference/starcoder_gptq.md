
### StarCoder


Run `TheBloke/starcoder-GPTQ` with TGI locally. 

```bash
# Model config
model=TheBloke/starcoder-GPTQ
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
  -v $volume:/data ghcr.io/huggingface/text-generation-inference:sha-e605c2a
```

send test request 

```bash
curl http://127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"\n    def test():\n        x=1+1\n        assert x ","parameters":{"max_new_tokens":60,"stop":["<|endoftext|>", "\n\n"],"top_p":0.95}}' \
    -H 'Content-Type: application/json'
```


load test with `k6`

```bash
k6 run starcoder_load.js
```

or with docker 
```bash
docker run --net=host -v $(pwd)/starcoder_load.js:/load.js loadimpact/k6:latest run /load.js
```


### Inference Results

We used `k6` with `constant-vus` executor, a fixed number of VUs execute as many iterations as possible for a specified amount of time.


| VU  | GPU  | time per token (p95) | queue time (p95) |
| --- | ---- | -------------------- | ---------------- |
| 1   | A10G | 30ms                 | 1ms              |
| 5   | A10G | 65ms                 | 105ms            |
| 10  | A10G | 104ms                | 120ms            |
| 20  | A10G | 203ms                | 5110ms           |
| 1   | A100 | 30ms                 | 1ms              |
| 5   | A100 | 59ms                 | 64ms               |
| 10  | A100 | 50ms                   | 51ms               |
| 20  | A100 | 59ms                   | 49ms               |
| 40  | A100 | 73ms                   | 1000ms               |
| 60  | A100 | 59ms                   | 113ms               |
| 80  | A100 | 92ms                   | 165ms               |
| 100  | A100 | 72ms                   | 1111ms               |
| 120  | A100 | 77ms                   | 1270ms               |
| 140  | A100 | _request start failing_                     | _request start failing_               |


