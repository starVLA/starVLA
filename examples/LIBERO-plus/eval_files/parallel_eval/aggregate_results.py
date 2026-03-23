import os
import json
import argparse
import glob

parser = argparse.ArgumentParser(description="aggregate results")

parser.add_argument("--root_path", help="")
args = parser.parse_args()
log_dir = args.root_path

task_suites = ['libero_10','libero_goal','libero_object','libero_spatial']
overall_results = {"overall": {'total_count':0, 'success_count': 0}}
for task_suite in task_suites:
    cur_root = os.path.join(log_dir, 'logs', task_suite)
    json_files = glob.glob(os.path.join(cur_root, "*.json"), recursive=True)
    for file in json_files:
        with open(file) as f:
            results = json.load(f)
        for item in results:
            overall_results["overall"]['total_count']+= results[item]['total_count']
            overall_results["overall"]['success_count']+= results[item]['success_count']
            if item not in overall_results:
                overall_results[item] = results[item]
            else:
                overall_results[item]['total_count'] += results[item]['total_count']
                overall_results[item]['success_count'] += results[item]['success_count']

for category in overall_results:
    overall_results[category]['success_rate'] = float(overall_results[category]['success_count']) / float(overall_results[category]['total_count'])

with open(os.path.join(log_dir,'overall_results.json'), 'w', encoding='utf-8') as f:
    json.dump(overall_results, f)