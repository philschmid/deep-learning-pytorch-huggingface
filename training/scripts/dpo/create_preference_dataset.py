from dataclasses import dataclass, field
import logging
import os
import time
from typing import cast
import re 

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from trl import TrlParser
from vllm import LLM, SamplingParams
from datasets import Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM

logger = logging.getLogger(__name__)

@dataclass
class CandidateArguments:
    generation_model_name_or_path: str = field(
        default=None,
        metadata={
            'help': 'Huggingface model name or path to model directory, for the model that will be used for generation, defaults to SFT model or previous iteration model.'
        },
    )
    dataset_id: str = field(
        default=None,
        metadata={
            'help': 'Path to the input dataset, that will be used to generate candidates, defaults to previous iteration output dataset.'
        },  
    )
    sample_size: int = field(
        default=None,
        metadata={
            'help': 'Number of samples to generate, defaults to as many as possible.'
        },
    )
    prompt_column: str = field(
        default='question',
        metadata={'help': 'Column name in the input dataset that contains the messages.'},
    )
    answer_column: str = field(
        default='answer',
        metadata={'help': 'Column name in the input dataset that contains the answer.'},
    )
    system_prompt: str = field(
        default= """Solve the given high school math problem by providing a clear explanation of each step leading to the final solution.
 
Provide a detailed breakdown of your calculations, beginning with an explanation of the problem and describing how you derive each formula, value, or conclusion. Use logical steps that build upon one another, to arrive at the final answer in a systematic manner.
 
# Steps
 
1. **Understand the Problem**: Restate the given math problem and clearly identify the main question and any important given values.
2. **Set Up**: Identify the key formulas or concepts that could help solve the problem (e.g., algebraic manipulation, geometry formulas, trigonometric identities).
3. **Solve Step-by-Step**: Iteratively progress through each step of the math problem, justifying why each consecutive operation brings you closer to the solution.
4. **Double Check**: If applicable, double check the work for accuracy and sense, and mention potential alternative approaches if any.
5. **Final Answer**: Provide the numerical or algebraic solution clearly, accompanied by appropriate units if relevant.
 
# Notes
 
- Always clearly define any variable or term used.
- Wherever applicable, include unit conversions or context to explain why each formula or step has been chosen.
- Assume the level of mathematics is suitable for high school, and avoid overly advanced math techniques unless they are common at that level.
""",
        metadata={'help': 'System prompt to use for generation.'},
    )
    num_solutions: int = field(
        default=5,
        metadata={'help': 'Number of solutions to generate for each input.'},
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Batch size for generation.'},
    )
    max_new_tokens: int = field(
        default=2048,
        metadata={'help': 'Maximum number of new tokens to generate.'},
    )
    temperature: float = field(
        default=0.7,
        metadata={'help': 'Temperature for generation.'},
    )
    top_p: float = field(
        default=1.0,
        metadata={'help': 'Top-p for generation.'},
    )

def score_solutions(
    candidate_result: str,
    ground_truth_result: str,
) -> bool:
    # finds the answer in the candidate result
    regex_pattern = r'\b\d+\b'
    match = re.findall(regex_pattern, candidate_result)
    
    if match:
        return match[-1]  == ground_truth_result
    else:
        return False
      

def vllm_create_candidates(
    dataset: Dataset,
    model_name_or_path: str,
    num_solutions: int,
    max_new_tokens: int,
    batch_size: int = 1,
    prompt_column: str = 'prompt',
    system_prompt: str = None,
    answer_column: str = 'answer',
    sample_size: int = None,
    **kwargs,
) -> Dataset:

    # Loads the model on all available GPUs with vLLM
    llm = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=4096,
    )
    # formats the prompt using the system prompt and the prompt column
    tokenizer = llm.get_tokenizer()
    def format_prompt(s):
        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": s[prompt_column]}
        ]
        return {"prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), "messages": messages}
    
    dataset = dataset.map(format_prompt)
    # print the first prompt
    print('First prompt:', dataset['prompt'][0])

    # set sampling params
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        n=num_solutions,
        temperature=kwargs.get('temperature', 1.0),
        top_p=kwargs.get('top_p', 1),
    )

    # Iterate over the dataset with batch size to generate candidates and create preference pairs based on the correct answer and ground truth
    preference_dataset = []
    for i in tqdm(range(0, len(dataset), batch_size), desc=f'Generating solutions: Already generated {len(preference_dataset)} preference pairs'):
        batch = dataset[i : i + batch_size]
        # Generate `num_solutions` candidates per batch
        result = llm.generate(batch['prompt'], sampling_params, use_tqdm=False)
        for j in range(0, len(batch['prompt'])):
            # iterate each candidate and check if it is correct
            preference_pair = {
                "system_prompt": system_prompt,
                "prompt": batch[prompt_column][j],
                "ground_truth": batch[answer_column][j],
            }
            for cand in result[j].outputs:
                # check if the candidate is correct
                cand_score = score_solutions(candidate_result=cand.text, ground_truth_result=batch[answer_column][j])                   
                if cand_score and preference_pair.get('chosen',None) is None:
                    preference_pair['chosen'] = cand.text
                elif not cand_score and preference_pair.get('rejected',None) is None:
                    preference_pair['rejected'] = cand.text
                # check if the pair is complete to prevent overwriting
                if preference_pair.get('chosen',None) and preference_pair.get('rejected',None):
                    continue
                
            # check is the generated candidates lead to a complete preference pair
            if preference_pair.get('chosen',None) and preference_pair.get('rejected',None):
                print(f'Found preference pair, adding to dataset.')
                preference_dataset.append(preference_pair)
        
        print(f'Generated {len(preference_dataset)} preference pairs')
        if len(preference_dataset) >= sample_size:
            break
    return Dataset.from_list(preference_dataset)


def main():
    parser = TrlParser((CandidateArguments))
    script_args = parser.parse_args_and_config()[0]
    script_args = cast(CandidateArguments, script_args)

    # load dataset and tokenizer
    dataset = load_dataset(script_args.dataset_id, split='train')
    print(f'Generating {script_args.num_solutions} solutions for {len(dataset)} prompts...')

    start_time = time.time()
    candidates_ds = vllm_create_candidates(
        dataset,
        model_name_or_path=script_args.generation_model_name_or_path,
        num_solutions=script_args.num_solutions,
        max_new_tokens=script_args.max_new_tokens,
        batch_size=script_args.batch_size,
        prompt_column=script_args.prompt_column,
        answer_column=script_args.answer_column,
        system_prompt=script_args.system_prompt,
        temperature=script_args.temperature,
        top_p=script_args.top_p,
        sample_size=script_args.sample_size if script_args.sample_size is not None else len(dataset),
    )
    print(f'Generated {len(dataset) * script_args.num_solutions} solutions in {time.time() - start_time:.2f} seconds.')
    
    save_dataset_id = f"{script_args.generation_model_name_or_path.replace('/', '-')[:40]}-{script_args.dataset_id.replace('/', '-')[:40]}-candidates"
    candidates_ds.push_to_hub(save_dataset_id)

if __name__ == '__main__':
    main()
