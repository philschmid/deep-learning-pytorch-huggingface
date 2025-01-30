from vllm import LLM, SamplingParams
from datasets import load_dataset
from random import randint

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

# use revision without "checkpoints-" as vLLM downloads all of them
llm = LLM(model="philschmid/qwen-2.5-3b-r1-countdown", revision="099c0f8cbfc522e7c3a476edfb749f576b164539")

# Load dataset from Hugging Face Hub
dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
dataset = load_dataset(dataset_id, split="train")
sample = dataset[randint(0, len(dataset))]  

# create conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
    {"role": "user", "content": f"Using the numbers {sample['nums']}, create an equation that equals {sample['target']}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."},
    {"role": "assistant", "content": "Let me solve this step by step.\n<think>"}
]
# generate response
res = llm.generate(llm.get_tokenizer().apply_chat_template(messages, tokenize=False, continue_final_message=True), sampling_params)
res = "<think>" + res[0].outputs[0].text
print(res)

# <think> We need to use the numbers 37, 15, 4, and 13 with basic arithmetic operations to make 16. Let's try different combinations:
# - 37 - 15 - 4 - 13 = 6 (too low)
# - 37 - 15 + 4 - 13 = 13 (too low)
# - 37 + 15 - 4 - 13 = 35 (too high)
# - 37 - 15 + 4 + 13 = 39 (too high)
# - 15 + 4 + 13 - 37 = -1 (too low)
# - 37 + 15 + 4 - 13 = 43 (too high)
# - 15 + 4 * 13 / 37 = 15 + 52 / 37 (not an integer)
# - 15 * 4 / 37 - 37 = -28.24 (not a whole number)
# - 4 * 13 / 15 - 37 = 41.3333 (not a whole number)
# After all combinations, I got not any integer result as 16.
# </think>
# <answer> 37 - 15 + 4 + 13 </answer>