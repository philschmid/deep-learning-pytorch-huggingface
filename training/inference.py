from vllm import LLM, SamplingParams
from datasets import load_dataset
from random import randint


sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

llm = LLM(model="/fsx/philipp/deep-learning-pytorch-huggingface/training/runs/qwen-r1-aha-moment")

# Load dataset from Hugging Face Hub
dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
dataset = load_dataset(dataset_id, split="train")


for i in range(10):
    sample = dataset[randint(0, len(dataset))]  
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
        {"role": "user", "content": f"Using the numbers {sample['nums']}, create an equation that equals {sample['target']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."},
    ]
    res = llm.generate(llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True), sampling_params)
    print(f"prompt: \n {messages[1]['content']}")
    print(f"target: \n {sample['target']}")
    print(f"response: \n {res[0].outputs[0].text}")