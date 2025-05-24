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
            model_dict[model] = [0] * 10
            number_dict[model] = [0] * 10
        for k, category in enumerate(['biological hazards', 'chemical hazards', 'radiation hazards', 'physical hazards',
                             'responsibility for safety', 'environmental and waste management',
                             'equipment usage', 'electricity safety', 'personal protective equipment (PPE)', 'emergency response']):
            for data_i in data:
                if category in data_i['Category']:
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
                model_dict[model] = [0] * 10
                number_dict[model] = [0] * 10
          for k, category in enumerate(['biological hazards', 'chemical hazards', 'radiation hazards', 'physical hazards',
                             'responsibility for safety', 'environmental and waste management',
                             'equipment usage', 'electricity safety', 'personal protective equipment (PPE)', 'emergency response']):
                for data_i in data:
                 if category in data_i['Category']:
                      number_dict[model][k] += 1
                      if data_i['Correct Answer'] == data_i['LLM Answer']:
                            model_dict[model][k] += 1

for model in model_dict:
    model_dict[model] = [round(model_dict[model][i] / number_dict[model][i], 4) * 100 for i in range(10)]

order = ['llama3-instruct', 'llama-3-70b', 'vicuna', 'vicuna-13b', 'mistral', 'mistral-8x7b', 'galactica',
         'instructBlip-7B', 'Qwen-VL-Chat', 'InternVL2', 'llama3_2', 'gemini-1.5-flash', 'gemini-1.5-pro',
         'claude3-haiku', 'claude3.5-sonnet', 'gpt-4o-mini', 'gpt-4o-2024-08-06']

model_dict = OrderedDict((key, model_dict[key]) for key in order)
print(number_dict)
# save to csv
import csv
with open('../../Logs/category_acc.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Biological hazards', 'Chemical hazards', 'Radiation hazards', 'Physical hazards',
                     'Responsibility for safety', 'Environmental and waste management', 'Equipment usage',
                     'Emergency response', 'Personal protective equipment (PPE)', 'Electricity safety'])
    for model in model_dict:
        writer.writerow([model] + model_dict[model])







