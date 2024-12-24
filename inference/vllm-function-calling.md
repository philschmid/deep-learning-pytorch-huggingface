# vLLM Function Calling Inference

This guide demonstrates how to run vLLM with function calling capabilities using Llama models.

```
docker run --gpus all \
    --env "HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)" \
    -p 8000:8000 \
    --shm-size=10G \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.3-70B-Instruct  --tensor-parallel-size 8 --max_model_len 4096  --enable-auto-tool-choice --tool-call-parser llama3_json 
```