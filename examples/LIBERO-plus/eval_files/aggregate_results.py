import os
import json
log_dir = os.environ.get('LOG_DIR')

task_suites = ['libero_10.json','libero_goal.json','libero_object.json','libero_spatial.json']
overall_results = {"overall": {'total_count':0, 'success_count': 0}}
for task_suite in task_suites:
    with open(os.path.join(log_dir,task_suite)) as f:
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