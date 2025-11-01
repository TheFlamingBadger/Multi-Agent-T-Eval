import argparse
import os
from typing import Dict, List, Optional

import mmengine
import numpy as np

np.set_printoptions(precision=1)

TOKEN_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    return parser.parse_args()


def aggregate_usage(trace) -> Dict[str, float]:
    """Aggregate token usage across a nested orchestration trace."""
    totals = {key: 0.0 for key in TOKEN_KEYS}
    found = False

    def _traverse(node):
        nonlocal found
        if isinstance(node, dict):
            usage = node.get('usage')
            if isinstance(usage, dict):
                for key in TOKEN_KEYS:
                    val = usage.get(key)
                    if isinstance(val, (int, float)):
                        totals[key] += float(val)
                        found = True
            for value in node.values():
                if isinstance(value, (dict, list)):
                    _traverse(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    _traverse(item)

    _traverse(trace)
    return totals if found else {}


def compute_file_stats(file_path: str) -> Optional[Dict[str, object]]:
    """Compute cumulative timing and token usage statistics for a result file."""
    if not os.path.exists(file_path):
        return None

    data = mmengine.load(file_path)
    time_sum = 0.0
    time_count = 0
    token_sums = {key: 0.0 for key in TOKEN_KEYS}
    token_count = 0

    iterable = data.items() if isinstance(data, dict) else enumerate(data)
    for _, entry in iterable:
        if not isinstance(entry, dict):
            continue

        inference_time = entry.get('inference_time_seconds')
        if isinstance(inference_time, (int, float)):
            time_sum += float(inference_time)
            time_count += 1

        usage_totals = aggregate_usage(entry.get('orchestration_trace'))
        if usage_totals:
            token_count += 1
            for key in TOKEN_KEYS:
                token_sums[key] += usage_totals.get(key, 0.0)

    return {
        'time_sum': time_sum,
        'time_count': time_count,
        'token_sums': token_sums,
        'token_count': token_count,
    }


def combine_category_stats(file_stats: List[Dict[str, object]]) -> Dict[str, object]:
    total_time = sum(stats['time_sum'] for stats in file_stats)
    total_time_count = sum(stats['time_count'] for stats in file_stats)
    combined_tokens = {
        key: sum(stats['token_sums'][key] for stats in file_stats) for key in TOKEN_KEYS
    }
    total_token_count = sum(stats['token_count'] for stats in file_stats)

    avg_time = (total_time / total_time_count) if total_time_count else None
    avg_tokens = {
        key: (combined_tokens[key] / total_token_count) if total_token_count else None
        for key in TOKEN_KEYS
    }

    return {
        'avg_time': avg_time,
        'avg_tokens': avg_tokens,
        'time_count': total_time_count,
        'token_count': total_token_count,
    }


def compute_scores(result):
    instruct_list = [
        (result['instruct_json']['json_format_metric'] + result['instruct_json']['json_args_em_metric']) / 2,
        (result['instruct_json']['string_format_metric'] + result['instruct_json']['string_args_em_metric']) / 2,
    ]
    plan_list = [result['plan_str']['f1_score'], result['plan_json']['f1_score']]
    reason_list = [result['reason_str']['thought'], result['rru_json']['thought']]
    retrieve_list = [result['retrieve_str']['name'], result['rru_json']['name']]
    understand_list = [result['understand_str']['args'], result['rru_json']['args']]
    review_list = [result['review_str']['review_quality'], result['review_str']['review_quality']]

    final_score = [
        np.mean(instruct_list),
        np.mean(plan_list),
        np.mean(reason_list),
        np.mean(retrieve_list),
        np.mean(understand_list),
        np.mean(review_list),
    ]
    overall = np.mean(final_score)
    final_score.insert(0, overall)
    return final_score


def derive_model_name(result_path: str) -> str:
    stem = os.path.splitext(os.path.basename(result_path))[0]
    return stem[:-3] if stem.endswith('_-1') else stem


def build_category_file_map(model_name: str) -> Dict[str, List[str]]:
    return {
        'Instruct': [f'instruct_{model_name}.json'],
        'Plan': [f'plan_str_{model_name}.json', f'plan_json_{model_name}.json'],
        'Reason': [f'reason_str_{model_name}.json', f'reason_retrieve_understand_json_{model_name}.json'],
        'Retrieve': [f'retrieve_str_{model_name}.json', f'reason_retrieve_understand_json_{model_name}.json'],
        'Understand': [f'understand_str_{model_name}.json', f'reason_retrieve_understand_json_{model_name}.json'],
        'Review': [f'review_str_{model_name}.json'],
    }


def format_value(value, decimals=2):
    return f"{value:.{decimals}f}" if isinstance(value, (int, float)) else "N/A"


def convert_results(result_path: str):
    result = mmengine.load(result_path)
    name_list = ['Overall', 'Instruct', 'Plan', 'Reason', 'Retrieve', 'Understand', 'Review']
    final_scores = compute_scores(result)

    print("Cut Paste Results: ", np.array(final_scores) * 100)
    for name, score in zip(name_list, final_scores):
        print(f"{name}: {score * 100:.1f}", end='\t')

    base_dir = os.path.dirname(result_path)
    model_name = derive_model_name(result_path)
    category_files = build_category_file_map(model_name)

    file_stats_cache: Dict[str, Optional[Dict[str, object]]] = {}
    category_stats: Dict[str, Dict[str, object]] = {}

    for category, filenames in category_files.items():
        stats_to_combine: List[Dict[str, object]] = []
        for filename in filenames:
            file_path = os.path.join(base_dir, filename)
            if file_path not in file_stats_cache:
                file_stats_cache[file_path] = compute_file_stats(file_path)
            file_stats = file_stats_cache[file_path]
            if file_stats:
                stats_to_combine.append(file_stats)
        if stats_to_combine:
            category_stats[category] = combine_category_stats(stats_to_combine)
        else:
            category_stats[category] = {
                'avg_time': None,
                'avg_tokens': {key: None for key in TOKEN_KEYS},
                'time_count': 0,
                'token_count': 0,
            }

    print("\n\nPer-category average inference time & token usage:")
    header = [
        "Category",
        "Avg Time (s)",
        "Avg Prompt Tokens",
        "Avg Completion Tokens",
        "Avg Total Tokens",
        "Time Samples",
        "Token Samples",
    ]
    print('\t'.join(header))

    for category in name_list[1:]:
        stats = category_stats.get(category, {})
        avg_time = format_value(stats.get('avg_time'))
        avg_prompt = format_value(stats.get('avg_tokens', {}).get('prompt_tokens'))
        avg_completion = format_value(stats.get('avg_tokens', {}).get('completion_tokens'))
        avg_total = format_value(stats.get('avg_tokens', {}).get('total_tokens'))
        time_samples = stats.get('time_count', 0)
        token_samples = stats.get('token_count', 0)

        print(
            f"{category}\t{avg_time}\t{avg_prompt}\t{avg_completion}\t"
            f"{avg_total}\t{time_samples}\t{token_samples}"
        )


if __name__ == '__main__':
    args = parse_args()
    convert_results(args.result_path)
