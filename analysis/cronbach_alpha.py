import json
import numpy as np


score_array = np.zeros((13, 632))
models = ['llama3-instruct', 'vicuna', 'mistral', 'vicuna-13b', 'llama-3-70b', 
          'mistral-8x7b', 'galactica', 'claude3-haiku', 'claude3.5-sonnet', 
          'gemini-1.5-flash', 'gemini-1.5-pro', 'gpt-4o-mini', 'gpt-4o-2024-08-06']

for model_idx, model in enumerate(models):
    log_dir = '../../Logs/' + model + '/'
    mode = 'DA'
    n_shot = '0'
    file_path = log_dir + f'final_QA_{mode}_{n_shot}_result.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, data_i in enumerate(data):
            if data_i['Correct Answer'] == data_i['LLM Answer']:
                score_array[model_idx][i] = 1


item_variances = np.var(score_array, axis=0, ddof=1)

total_scores = np.sum(score_array, axis=1)
total_variance = np.var(total_scores, ddof=1)
num_items = score_array.shape[1]
alpha = (num_items / (num_items - 1)) * (1 - (np.sum(item_variances) / total_variance))
print(f"Cronbach's Alpha: {alpha:.2f}")

