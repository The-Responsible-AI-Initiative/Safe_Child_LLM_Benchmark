import json
import numpy as np
from collections import OrderedDict

model_dict = {}
number_dict = {}
for model in ['llama3-instruct', 'vicuna', 'mistral', 'vicuna-13b', 'llama-3-70b', 'mistral-8x7b', 'galactica',
              'claude3-haiku', 'claude3.5-sonnet', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gpt-4o-mini', 'gpt-4o-2024-08-06']:
    log_dir = '../../Logs/' + model + '/'
    mode = 'DA'
    n_shot = '0'
    file_path = log_dir + f'final_QA_{mode}_{n_shot}'
    file_path += '_result.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if model not in model_dict:
            model_dict[model] = [0] * 2
            number_dict[model] = [0] * 2
        for k, category in enumerate(['Easy', 'Hard']):
            for data_i in data:
                if data_i['Level'] == category:
                    number_dict[model][k] += 1
                    if data_i['Correct Answer'] == data_i['LLM Answer']:
                        model_dict[model][k] += 1

for model in ['instructBlip-7B', 'Qwen-VL-Chat', 'InternVL2', 'llama3_2', 'claude3-haiku', 'claude3.5-sonnet',
                'gemini-1.5-flash', 'gemini-1.5-pro', 'gpt-4o-mini', 'gpt-4o-2024-08-06']:
     log_dir = '../../Logs/' + model + '/'
     mode = 'DA'
     n_shot = '0'
     file_path = log_dir + f'final_QA_I_{mode}_0'
     file_path += '_result_I.json'
     with open(file_path, 'r', encoding='utf-8') as f:
          data = json.load(f)
          if model not in model_dict:
                model_dict[model] = [0] * 2
                number_dict[model] = [0] * 2
          for k, category in enumerate(['Easy', 'Hard']):
                for data_i in data:
                     if data_i['Level'] == category:
                          number_dict[model][k] += 1
                          if data_i['Correct Answer'] == data_i['LLM Answer']:
                              model_dict[model][k] += 1

print(model_dict)
print(number_dict)
for model in model_dict:
    model_dict[model] = [round(model_dict[model][i] / number_dict[model][i], 4) * 100 for i in range(2)]

order = ['llama3-instruct', 'llama-3-70b', 'vicuna', 'vicuna-13b', 'mistral', 'mistral-8x7b', 'galactica',
         'instructBlip-7B', 'Qwen-VL-Chat', 'InternVL2', 'llama3_2', 'gemini-1.5-flash', 'gemini-1.5-pro',
         'claude3-haiku', 'claude3.5-sonnet', 'gpt-4o-mini', 'gpt-4o-2024-08-06']

model_dict = OrderedDict((key, model_dict[key]) for key in order)
print(number_dict)
# save to csv
import csv
with open('../../Logs/level_acc.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Easy', 'Hard'])
    for model in model_dict:
        writer.writerow([model] + model_dict[model])







