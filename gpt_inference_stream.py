from openai import OpenAI
import os
from tqdm import tqdm
import time, json
import multiprocessing
import argparse
import glob

import logging
import sys
from parser import load_jsonl
from evaluate import evaluate_result
from evaluate_sampling import evaluate_result as evaluate_result_sampling

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s-%(name)s-[%(levelname)s]-[%(message)s]")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("start...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--final_filename', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--short_model', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--base_url', type=str, required=True)
    parser.add_argument('--api_key_file', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--thinking', type=int, choices=[0, 1], default=0)
    parser.add_argument('--cot_type', type=str, choices=['zero', 'raw', 'zero_budget'], default='raw')
    parser.add_argument('--T', type=float)
    parser.add_argument('--top_p', type=float)
    parser.add_argument('--N', type=int)

    return parser.parse_args()


def load_api(path: str):
    api_keys = []
    with open(path, 'r') as f:
        for line in f:
            key = line.strip()
            api_keys.append(key)
    return api_keys


def load_file(path):
    finished_ids = []
    with open(path, 'r') as f:
        for line in f.readlines():
            finished_ids.append(json.loads(line)['id'])
    return finished_ids

def get_system_prompt(answer_type):
    if answer_type == 'yes_no':
        return "You are a helpful assistant. Answer the following question and output the answer in the format: ```\\box {Your Answer}\n```. Make sure that the answer is either 'YES' or 'NO'. Question:"
    elif answer_type == 'numeric':
        return "You are a helpful assistant. Answer the following question and output the answer in the format: ```\\box {Your Answer}\n```. Make sure that the answer is a number. Question:"
    elif answer_type == 'structured_tuple':
        return "You are a helpful assistant. Answer the following question and output the answer in the format: ```\\box {Your Answer}\n```. Make sure that the answer is a structured tuple. Question:"
    else:
        return "You are a helpful AI assistant. Your goal is to solve the following problem and provide the final answer. The final answer must be enclosed within a ```\\box{}\n``` command. For example: ```\\box{Your Final Answer}\n```."


def get_user_prompt(cot_type, prompt):
    if cot_type == 'zero':
        prompt = prompt + '\nLet\'s think step by step.'
    elif cot_type == 'zero_budget':
        prompt = prompt + '\nLet\'s think step by step and try your best to think more than 10 steps.'

    return prompt


def gpt_completion(item):
    idx, args, prompt_block, api_key, output_path = item
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    if os.path.exists(output_path):
        finished_ids = load_file(output_path)
    else:
        finished_ids = []
    output_f = open(output_path, 'a')
    print(f'Worker {idx} start', 'total:', len(prompt_block), 'finished:', len(finished_ids))
    for sample in tqdm(prompt_block, total=len(prompt_block), desc=f'Worker {idx}'):
        sample = json.loads(sample)
        prompt = sample['question']
        task_id = sample['id']
        if task_id in finished_ids:
            continue

        sample['completions'] = []
        sample['time'] = []
        zero_params = dict(
            model=args.model,
            messages=[
                {"role": "system", "content": get_system_prompt(None)},
                {'role': 'user', 'content': get_user_prompt(args.cot_type, prompt)}
            ],
            temperature=args.T, max_tokens=args.max_tokens,
            # extra_body={
            #     "reasoning": {"enabled": True}
            # }
        )
        above_zero_params = zero_params.copy()
        above_zero_params['top_p'] = args.top_p
        while len(sample['completions']) < args.N:
            flag = False
            while not flag:
                time_begin = time.time()
                try:
                    if args.T == 0:
                        response = client.chat.completions.create(**zero_params)
                    elif args.T > 0:
                        response = client.chat.completions.create(**above_zero_params)
                    flag = True
                except Exception as e:
                    print(f'Worker {idx}', e)
                    time.sleep(2)

                time_end = time.time()

            for choice in response.choices:
                assert choice.message.role == 'assistant'
                if getattr(choice.message, 'reasoning', None):
                    sample['completions'].append(choice.message.reasoning + choice.message.content)
                else:
                    sample['completions'].append(choice.message.content)

                sample['time'].append(time_end - time_begin)

            time.sleep(2)

        # del sample['prompt']
        output_f.write(json.dumps(sample) + '\n')
        output_f.flush()

    output_f.close()


def gpt_completion_stream(item):
    idx, args, prompt_block, api_key, output_path = item
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    if os.path.exists(output_path):
        finished_ids = load_file(output_path)
    else:
        finished_ids = []
    output_f = open(output_path, 'a')
    print(f'Worker {idx} start', 'total:', len(prompt_block), 'finished:', len(finished_ids))

    for sample in tqdm(prompt_block, total=len(prompt_block), desc=f'Worker {idx}'):
        prompt = sample['question']
        task_id = sample['id']
        if task_id in finished_ids:
            continue
        user_prompt = get_user_prompt(args.cot_type, prompt)
        sample['completions'] = []
        sample['time'] = []
        if args.thinking == 1:
            extra_body = {
                "reasoning": {"enabled": True}
            }
        else:
            extra_body = {}

        # print(f"======= {task_id} ========\n{user_prompt}")
        while len(sample['completions']) < args.N:
            flag = False
            time_begin = time.time()
            retry_times = 0
            zero_params = dict(
                model=args.model,
                messages=[
                    {"role": "system", "content": get_system_prompt(None)},
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=args.T, max_tokens=args.max_tokens, stream=True,
                extra_body=extra_body
            )
            above_zero_params = zero_params.copy()
            above_zero_params['top_p'] = args.top_p
            stream = None
            while not flag and retry_times < 3:
                time_begin = time.time()
                try:
                    if args.T == 0:
                        stream = client.chat.completions.create(**zero_params)
                    elif args.T > 0:
                        stream = client.chat.completions.create(**above_zero_params)
                    flag = True
                except KeyboardInterrupt as e:
                    print("Error occured:", e)
                    exit(1)
                except Exception as e:
                    print(f'Worker {idx}', e)
                    time.sleep(2)

            if not stream:
                print(f'Worker {idx} stream is None')
                continue

            completion = {}
            reasoning = {}
            try:
                for chunk in stream:
                    for choice in chunk.choices:
                        delta = choice.delta
                        index = choice.index

                        if delta.content or getattr(delta, 'reasoning', None):
                            if getattr(delta, 'reasoning', None):
                                completion[index] = completion.get(index, '') + delta.reasoning + delta.content
                                reasoning[index] = reasoning.get(index, '') + delta.reasoning
                            else:
                                completion[index] = completion.get(index, '') + delta.content
                        # print(f'Worker {idx} {task_id}', len(completion.get(index, '')))

            except Exception as e:
                error_info = f"FATAL ERROR {idx} on task_id {sample.get('id', 'unknown')}: {e}"
                logger.info(error_info)
                continue

            print("reasoning: ", reasoning)
            assert hasattr(chunk, 'usage') and chunk.usage is not None, f'[{task_id}] {chunk}'
            time_end = time.time()
            sample['reasoning'] = reasoning[0]
            for index, content in completion.items():
                sample['completions'].append(content)
                sample['time'].append(time_end - time_begin)

            # record token number
            sample['usage'] = {
                'prompt_length': chunk.usage.prompt_tokens,
                'response_length': chunk.usage.completion_tokens
            }

            time.sleep(1)

        # del sample['prompt']
        output_f.write(json.dumps(sample) + '\n')
        output_f.flush()
    output_f.close()


def merge(output_dir, final_filename, delete_parts):
    """
    Merges all 'completion_block*.jsonl' files into a single file.
    """
    print("\n--- Starting merge process ---")

    # Construct the path for the final merged file
    final_filepath = os.path.join(output_dir, final_filename)

    # Find all partial files using a glob pattern
    partial_files = sorted(glob.glob(os.path.join(output_dir, 'completion_block*.jsonl')))
    # assert partial_files, "No partial files found to merge."
    if not partial_files:
        print("No partial files found to merge.")
        return

    all_items = []
    for filename in partial_files:
        partial_items = load_jsonl(filename)
        all_items.extend(partial_items)

    with open(final_filepath, 'w', encoding='utf-8') as outfile:
        for item in all_items:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nSuccessfully merged all parts into: {final_filepath}")

    # Optionally, clean up by deleting the partial files
    if delete_parts:
        print("Deleting partial files...")
        for filename in partial_files:
            os.remove(filename)
        print("Partial files deleted.")

    print("--- Merge process finished ---")

if __name__ == "__main__":
    args = parse_args()
    # step1. query llm
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    api_pool = load_api(args.api_key_file)

    print("[args] ", vars(args))

    if args.mode == 'greedy':
        args.T = 0
        args.top_p = None
        args.N = 1

    problems = load_jsonl(args.prompt_file)

    task_block = []
    if len(problems) % len(api_pool) == 0:
        l = len(problems) // len(api_pool)
        pool_num = len(api_pool)
    else:
        l = len(problems) // len(api_pool) + 1
        pool_num = len(problems) // l + 1 if len(problems) % l != 0 else len(problems) // l

    for i in range(pool_num):
        if i == pool_num - 1:
            prompt_block = problems[i*l:]
        else:
            prompt_block = problems[i*l:(i+1)*l]
        api_key = api_pool[i]
        output_path = f'{args.output_dir}/completion_block{i}.jsonl'
        task_block.append((i, args, prompt_block, api_key, output_path))

    pool = multiprocessing.Pool(pool_num)
    pool.map(gpt_completion_stream, task_block)
    pool.close()
    pool.join()

    # step2. merge
    merge(args.output_dir, args.final_filename, delete_parts=True)

    # step3. evaluate
    completions = load_jsonl(os.path.join(args.output_dir, args.final_filename))
    result_filename = os.path.join(args.output_dir, f'{args.short_model}_result.jsonl')
    print(f"result_filename is {result_filename}")
    if args.mode == 'greedy':
        evaluate_result(args.prompt_file, completions, result_filename, args.short_model)
    elif args.mode == 'sampling':
        evaluate_result_sampling(args.prompt_file, completions, result_filename, args.short_model)
    print("evaluate finished")
