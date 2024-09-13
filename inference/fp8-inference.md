# Benchmark and compare FP8 and FP16 inference for vLLM

[vLLM supports FP8](https://docs.vllm.ai/en/latest/quantization/fp8.html) (8-bit floating point) weight and activation quantization using hardware acceleration on GPUs such as Nvidia H100 and AMD MI300x. Currently, only Hopper and Ada Lovelace GPUs are officially supported for W8A8. We are using `guidellm`

## Summary

**Model Memory Usage**
- FP8: 8.49 GB
- FP16: 14.99 GB
- Memory Savings: ~43%


**Performance Highlights**
- Max Requests per Second:
  - FP8: 2.54 req/sec at rate 8/128
  - FP16: 1.37 req/sec at rate 2
  - Improvement: ~85%
- Token Throughput:
  - FP8: 587.68 tokens/sec at rate 64
  - FP16: 302.94 tokens/sec at rate 2
  - Improvement: ~94%

- Request Latency:
  - FP8: 12.26 sec at rate 1
  - FP16: 21.87 sec at rate 1
  - Improvement: ~44%

**Results:**
- FP8 consistently outperforms FP16 across all metrics at the same concurrency level.
- FP8 shows the most significant improvement in Request Latency.
- Even at higher concurrency levels, FP8 generally maintains better performance (though direct comparisons at other levels should be made carefully).


## FP8 Inference

1. run vLLM with FP8

```bash
docker run --gpus all \
    -p 8080:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic --max_model_len 8192
```
Note: Loading model weights took 8.4939 GB


2. benchmark with `guidellm`

```bash
guidellm \
  --target "http://localhost:8080/v1" \
  --model "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic" \
  --data-type emulated \
  --data "prompt_tokens=550,generated_tokens=250" \
  --rate-type constant --rate 1 --rate 2 --rate 4 --rate 8 --rate 16 --rate 64 --rate 128 \
  --max-seconds 90
```

```bash
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓ 
┃ Benchmark                   ┃ Requests per Second ┃ Request Latency ┃ Time to First Token ┃ Inter Token Latency ┃ Output Token Throughput ┃ 
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩ 
│ asynchronous@1.00 req/sec   │ 0.88 req/sec        │ 12.26 sec       │ 240.18 ms           │ 54.61 ms            │ 194.47 tokens/sec       │
│ asynchronous@2.00 req/sec   │ 1.32 req/sec        │ 16.43 sec       │ 267.88 ms           │ 72.65 ms            │ 294.87 tokens/sec       │
│ asynchronous@4.00 req/sec   │ 1.53 req/sec        │ 47.30 sec       │ 19242.07 ms         │ 127.31 ms           │ 338.21 tokens/sec       │
│ asynchronous@8.00 req/sec   │ 2.54 req/sec        │ 31.57 sec       │ 3144.09 ms          │ 124.14 ms           │ 582.76 tokens/sec       │
│ asynchronous@16.00 req/sec  │ 2.26 req/sec        │ 58.66 sec       │ 29508.54 ms         │ 127.98 ms           │ 516.97 tokens/sec       │
│ asynchronous@64.00 req/sec  │ 2.49 req/sec        │ 39.48 sec       │ 9327.19 ms          │ 127.77 ms           │ 587.68 tokens/sec       │
│ asynchronous@128.00 req/sec │ 2.54 req/sec        │ 37.21 sec       │ 10749.84 ms         │ 118.26 ms           │ 569.52 tokens/sec       │
└─────────────────────────────┴─────────────────────┴─────────────────┴─────────────────────┴─────────────────────┴─────────────────────────┘ 
```


## FP16 Inference

1. run vLLM with FP16

```bash
docker run --gpus all \
    -p 8080:8000 \
    --ipc=host \
    --env "HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)" \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --max_model_len 8192
```

Note: Loading model weights took 14.99 GB 

1. benchmark with `guidellm`

```bash
guidellm \
  --target "http://localhost:8080/v1" \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --data-type emulated \
  --data "prompt_tokens=550,generated_tokens=250" \
  --rate-type constant --rate 1 --rate 2 --rate 4 --rate 8 --rate 16 --rate 64 --rate 128 \
  --max-seconds 90
```

```bash
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓ 
┃ Benchmark                   ┃ Requests per Second ┃ Request Latency ┃ Time to First Token ┃ Inter Token Latency ┃ Output Token Throughput ┃ 
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩ 
│ asynchronous@1.00 req/sec   │ 0.76 req/sec        │ 21.87 sec       │ 314.05 ms           │ 94.95 ms            │ 172.09 tokens/sec       │
│ asynchronous@2.00 req/sec   │ 1.37 req/sec        │ 23.48 sec       │ 820.36 ms           │ 102.23 ms           │ 302.94 tokens/sec       │
│ asynchronous@4.00 req/sec   │ 1.02 req/sec        │ 45.64 sec       │ 19181.45 ms         │ 118.46 ms           │ 228.36 tokens/sec       │
│ asynchronous@8.00 req/sec   │ 0.94 req/sec        │ 49.13 sec       │ 23194.74 ms         │ 115.74 ms           │ 211.55 tokens/sec       │
│ asynchronous@64.00 req/sec  │ 0.89 req/sec        │ 56.25 sec       │ 30167.99 ms         │ 115.69 ms           │ 199.90 tokens/sec       │
│ asynchronous@16.00 req/sec  │ 1.25 req/sec        │ 56.19 sec       │ 31740.33 ms         │ 106.55 ms           │ 285.55 tokens/sec       │
│ asynchronous@128.00 req/sec │ 1.00 req/sec        │ 53.18 sec       │ 27422.15 ms         │ 113.62 ms           │ 225.60 tokens/sec       │
└─────────────────────────────┴─────────────────────┴─────────────────┴─────────────────────┴─────────────────────┴─────────────────────────┘ 
```