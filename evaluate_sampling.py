
import json
import re

import os
import ast
import math
import re
import shutil
from collections import Counter
import numpy as np
import pandas as pd
import itertools
from parser import are_equivalent, are_equivalent_of_ratios, load_jsonl, get_answer_n, extract_answer


from collections import defaultdict
from parser import get_order

def estimate_pass_at_k(
        num_samples, correct_arr, k
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    return estimator(int(num_samples), int(correct_arr[0]), k)

def calculate_majority(problems):
    data_list = []
    for p in problems:
        order = get_order(p)
        row = {
            'id': p['id'],
            'order': order,
            'pass': int(p['majority_pass']),
            'pass_ratio': p['majority_pass_ratio']
        }

        data_list.append(row)
        total_row = row.copy()
        total_row['order'] = -1
        data_list.append(total_row)

    df = pd.DataFrame(data_list)
    grouped = df.groupby(['order'])[['pass', 'pass_ratio']].mean()
    reshaped = grouped.unstack()
    reshaped = reshaped.reorder_levels([1, 0]).sort_index(ascending=[False, True])
    result_df = reshaped.to_frame().T 
    result_df.columns = [f"order{col[0]}_{col[1]}" for col in result_df.columns]
    return result_df


def calculate_metrics(problems):
    # table 5
    ks = [1, 2, 4, 8, 16, 32]

    data_list = []
    for k in ks:
        for p in problems:
            order = get_order(p)
            row = {
                'k': k,
                'id': p['id'],
                'order': order,
                'pass': float(p['pass_at_k'][k - 1]),
                'pass_ratio': float(p['pr_pass_at_k'][k - 1])
            }

            data_list.append(row)
            total_row = row.copy()
            total_row['order'] = -1
            data_list.append(total_row)

    df = pd.DataFrame(data_list)

    grouped = df.groupby(['k', 'order'])[['pass', 'pass_ratio']].mean()
    print(grouped)

    # Move the 'order' level of the index to the columns
    reshaped = grouped.unstack(level='order')

    # Reorder the columns to group 'pass' and 'pass_ratio' belonging to the same 'order' together
    reshaped = reshaped.reorder_levels(['order', None], axis=1).sort_index(axis=1)

    reshaped = reshaped.sort_index(axis=1, level=[0, 1], ascending=[False, True])
    reshaped.columns = [f"{col[0]}_{col[1]}" for col in reshaped.columns]
    result = reshaped.reset_index()
    print(reshaped.columns)
    return result

def get_total_mp(items):
    total_mp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for problem in items:
        order = get_order(problem)
        total_mp[order] += 1
    return total_mp


def get_pass_ratio_pass_at_k(answer_matrix, k):
    """
    Parameter:
    answer_matrix: List[List[int]], shape is [n_problem, sampling_count]
    k: sampling count for each problem
    """
    total_pass_at_k = 0.0
    num_questions = len(answer_matrix)

    if num_questions == 0:
        return 0.0

    for row in answer_matrix:
        n = len(row)
        c = sum(row)
        if n - c < k:
            total_pass_at_k += 1.0
        else:
            negative_cases = math.comb(n - c, k)
            total_cases = math.comb(n, k)

            prob = 1.0 - (negative_cases / total_cases)
            total_pass_at_k += prob

    return total_pass_at_k / num_questions

def evaluate_result(dataset_path, problems, output_path, model):
    exception_count = 0
    correct_count_m = 0
    correct_count_p8 = 0
    correct_count_p32 = 0
    sampling_count = len(problems[0]['completions'])

    with open(output_path, 'w', encoding='utf-8') as f_out:
        pass_ratio_count = 0.0
        correct_dict = defaultdict(int)
        pass_ratio_dict = defaultdict(float)
        for problem in problems:
            results = []
            flags = []
            pass_ratios = []
            order = get_order(problem)
            answer_n = get_answer_n(problem['correct_answer'])
            answer_matrix = [[0] * sampling_count for i in range(answer_n)]

            for idx in range(sampling_count):
                flag = False
                pass_ratio = 0.0
                result = extract_answer(problem['completions'][idx], model)
                problem['correct_answer'] = problem['correct_answer'].replace('\'', '\"')
                if result:
                    v1, v2, equal, pass_ratio_list = are_equivalent_of_ratios(result, problem['correct_answer'])
                    # v1, v2, equal, pass_ratio = are_equivalent(result, problem['correct_answer'])
                    pass_ratio = sum(pass_ratio_list) / answer_n
                    flag = bool(equal)
                    pass_ratio_count += pass_ratio
                    correct_dict[order] += int(flag)
                    pass_ratio_dict[order] += pass_ratio
                    answer_range = min(len(pass_ratio_list), answer_n)
                    for answer_idx in range(answer_range):
                        answer_matrix[answer_idx][idx] = pass_ratio_list[answer_idx]
                else:
                    exception_count += 1

                results.append(result)
                flags.append(flag)
                pass_ratios.append(pass_ratio)

            # majority vote and pass@k
            result_counter = Counter(results)
            majority_result, majority_count = result_counter.most_common(1)[0]
            max_count_idx = results.index(majority_result)

            majority_flag = flags[max_count_idx]
            majority_pass_ratio = pass_ratios[max_count_idx]
            if majority_flag:
                correct_count_m += 1

            # calculate pass@k
            correct_arr = [sum(flags)]
            pass_at_k = [estimate_pass_at_k(sampling_count, correct_arr, k) for k in range(1, sampling_count + 1)]
            pr_pass_at_k = [get_pass_ratio_pass_at_k(
                            answer_matrix, k) for k in range(1, sampling_count + 1)]
            # statis for sampling
            correct_count_p8 += pass_at_k[7]
            correct_count_p32 += pass_at_k[31]
            print("order", problem['id'], [sum(row) for row in answer_matrix], len(pr_pass_at_k))

            f_out.write(json.dumps({
                "id": problem['id'],
                "correct_answer": problem['correct_answer'],
                "result": results,
                "pass": flags,
                "pass_ratio": pass_ratios,
                "majority_result": majority_result,
                "majority_pass": majority_flag,
                "majority_pass_ratio": majority_pass_ratio,
                "pass_at_k": pass_at_k,
                "pr_pass_at_k": pr_pass_at_k
            }) + "\n")

    # count for each order
    total_mp = get_total_mp(problems)

    for k, v in correct_dict.items():
        print('Category{}'.format(k))
        print('Accuracy：{:.2f}， PassRatio：{:.2f}'.format(correct_dict[k] / total_mp[k] * 100, pass_ratio_dict[k] / total_mp[k] * 100))

    total_count = len(problems)
    print("Model: {}".format(model))
    print("Total: {}".format(total_count))
    print("Sampling Count: {}".format(sampling_count))
    print("Model: {}".format(model))
    print("Total: {}".format(total_count))
    print('Failed to extract answers：', exception_count)
    print('Majority Accuracy', correct_count_m / total_count * 100 if problems else 0)
    print('Pass@8 Accuracy：', correct_count_p8 / total_count * 100 if problems else 0)
    print('Pass@32 Accuracy：', correct_count_p32 / total_count * 100 if problems else 0)
    print('PassRatio：', pass_ratio_count / total_count * 100 if lines else 0)
    return

def get_all_models(directory):
    files = []
    for file in os.listdir(directory):
        files.append(file)

    models = []
    for file in files:
        match = re.match(r'^(.*?)_completion.jsonl$', file)
        if match:
            models.append(match.group(1))
    return models

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    models = get_all_models(args.input_dir)

    import pandas as pd
    pd.options.display.float_format = '{:.3f}'.format
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None
    pd.options.display.width = 1000
    # table 5
    for model in models:
        print('=' * 40)
        print(model)
        input_path = os.path.join(args.input_dir, f'{model}_completion.jsonl')
        output_path = os.path.join(args.input_dir, f'{model}_result.jsonl')

        lines = load_jsonl(input_path)

        print("results", len(lines))
        evaluate_result(input_path, lines, output_path, model)
        problems = load_jsonl(output_path)
        group = calculate_metrics(problems)
        print("len {}".format(len(problems)))
        print("best-of-n")
        print(group)
        majority_df = calculate_majority(problems)
        print("majority")
        print(majority_df)

# python evaluate_sampling.py --input_dir ./test_time_scaling