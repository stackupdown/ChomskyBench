
import json
import re

import os
import ast
import re
import shutil
from collections import Counter
import numpy as np
import itertools
import pandas as pd
from parser import are_equivalent, are_equivalent_of_ratios, get_answer_n, dump_jsonl, load_jsonl, extract_answer
import math
from collections import defaultdict


def get_order(s):
    res = -1
    mp = {'re_': 0, 'csg': 1, 'cfg': 2, 'dcfg': 3, 'regular_': 4}
    for k, v in mp.items():
        if s['id'].startswith(k):
            res = v
            break
    return res


def get_total_mp(items):
    total_mp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for problem in items:
        order = get_order(problem)
        total_mp[order] += 1
    return total_mp

def evaluate_result(dataset_path, lines, output_path, model):
    exception_count = 0
    correct_count_m = 0
    sampling_count = len(lines[0]['completions'])

    pass_ratio_count = 0.0
    correct_dict = defaultdict(int)
    pass_ratio_dict = defaultdict(float)
    problem_metrics = []
    for problem in lines:
        results = []
        flags = []
        pass_ratios = []
        # For each iteration
        sampling_count = len(problem['completions'])
        answer_n = get_answer_n(problem['correct_answer'])
        for idx in range(sampling_count):
            flag = False
            pass_ratio = 0.0
            result = extract_answer(problem['completions'][0], model)
            problem['correct_answer'] = problem['correct_answer'].replace('\'', '\"')
            if result:
                v1, v2, equal, pass_ratio_list = are_equivalent_of_ratios(result, problem['correct_answer'])
                flag = bool(equal)
                pass_ratio = sum(pass_ratio_list) / answer_n
                order = get_order(problem)
                correct_dict[order] += int(flag)
                pass_ratio_dict[order] += pass_ratio

            results.append(result)
            flags.append(flag)
            pass_ratios.append(pass_ratio)

        # Calculate majority result and pass@k
        if results:
            result_counter = Counter(results)
            majority_result, majority_count = result_counter.most_common(1)[0]
            max_count_idx = results.index(majority_result)
        else:
            max_count_idx = 0

        majority_flag = flags[max_count_idx]
        majority_pass_ratio = pass_ratios[max_count_idx]
        if majority_flag:
            correct_count_m += 1
        problem_metrics.append({
            'id': problem['id'],
            'order': get_order(problem),
            'model': model,
            'pass': flags[0],
            'pass_ratio': pass_ratios[0],
            "correct_answer": problem['correct_answer'],
            "result": results[0],
        })

    dump_jsonl(problem_metrics, output_path)
    grammar_list = []
    total_mp = get_total_mp(lines)
    print(total_mp)

    correct_count = 0
    for k in range(4, -1, -1):
        v = correct_dict[k]
        print('Category: {}'.format(k))
        print('Accuracy：{:.2f}% ({}/{})，PassRatio：{:.2f}% ({:.2f}/{})'.format(
            correct_dict[k] / total_mp[k] * 100, correct_dict[k], total_mp[k],
            pass_ratio_dict[k] / total_mp[k] * 100, pass_ratio_dict[k], total_mp[k]
        ))
        grammar_list.append({
            "order": k,
            "correct": correct_dict[k],
            "total": total_mp[k],
            "model": model,
            "acc": correct_dict[k] / total_mp[k],
            "pr": pass_ratio_dict[k] / total_mp[k],
        })
        correct_count += correct_dict[k]

    total_count = len(lines)
    print("Model: {}".format(model))
    print("Total: {}".format(total_count))
    print('Failed to extract answers：', exception_count)
    print('Wrong answers：', total_count - correct_count - exception_count)
    print('Correct answers：', correct_count)
    pass_ratio_count = sum([v for v in pass_ratio_dict.values()])

    # avg append
    grammar_list.append({
        "order": -1,
        "correct": correct_count,
        "total": total_count,
        'acc': correct_count / total_count,
        "model": model,
        'pr': pass_ratio_count / total_count if lines else 0
    })
    print('PassRatio：', pass_ratio_count / total_count * 100 if lines else 0)
    return problem_metrics, grammar_list


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


def print_table_acc(df):
    df_grouped = df.groupby(['model', 'order'])[['acc', 'pr']].first().reset_index()

    # 2. convert to wide table
    df_pivot = df_grouped.pivot(index='model', columns='order', values=['acc', 'pr'])

    desired_order = [4, 3, 2, 1, 0, -1]
    existing_orders = [o for o in desired_order if o in df_pivot.columns.get_level_values(1)]

    new_columns = []
    for o in existing_orders:
        new_columns.append(('acc', o))
        new_columns.append(('pr', o))

    df_final = df_pivot.reindex(columns=new_columns)

    name_map = {'0': 're', '1': 'cs', '2': 'ncf', '3': 'dcf', '4': 'regular', '-1': 'avg'}
    df_final.columns = [f"{name_map.get(str(o), o)}/{metric}" for metric, o in df_final.columns]
    return df_final


def draw_fig5(df):
    # figure 5 in essay
    df_grouped = df.groupby(['model', 'order'])[['acc']].first().reset_index()

    df_pivot = df_grouped.pivot(index='model', columns='order', values=['acc'])

    desired_order = [4, 3, 2, 1, 0, -1]
    existing_orders = [o for o in desired_order if o in df_pivot.columns.get_level_values(1)]

    new_columns = []
    for o in existing_orders:
        new_columns.append(('acc', o))

    df_final = df_pivot.reindex(columns=new_columns)
    name_map = {'0': 're', '1': 'cs', '2': 'ncf', '3': 'dcf', '4': 'regular', '-1': 'avg'}
    df_final.columns = [name_map.get(str(o)) for metric, o in df_final.columns]
    df_final = df_final.reset_index()
    return df_final


if __name__ == '__main__':
    # for table 3 in essay (acc and pass_ratio for models)

    args = parse_args()
    models = get_all_models(args.input_dir)
    total_list = []
    total_metrics = []
    for model in models:
        print('=' * 40)
        print(model)
        input_path = os.path.join(args.input_dir, f'{model}_completion.jsonl')
        output_path = os.path.join(args.input_dir, f'{model}_result.jsonl')

        lines = load_jsonl(input_path)
        print("result length", len(lines))

        problem_metrics, grammar_list = evaluate_result(input_path, lines, output_path, model)
        total_list.extend(grammar_list)
        total_metrics.extend(problem_metrics)

    pd.options.display.float_format = '{:.3f}'.format
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None
    pd.options.display.width = 1000

    df = pd.DataFrame(total_list)
    df.sort_values(by=['model', 'order'], inplace=True)
    df = df.round(3)

    df_final = print_table_acc(df)
    df_final = df_final.reindex(models).reset_index()
    print(df_final)
