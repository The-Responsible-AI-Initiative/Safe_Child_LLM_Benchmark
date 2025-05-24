import sys
import os
import time
import traceback

# Get the current directory (src/tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the src directory path
src_dir = os.path.abspath(os.path.join(current_dir, '../'))

# Add src directory to Python path
sys.path.insert(0, src_dir)
import json
import asyncio
import tiktoken
from pathlib import Path
from openai import AsyncOpenAI

# Initialize OpenAI client
client = AsyncOpenAI()

# Get model list from config.py (example)
async_models = ['deepseek-r1']
local_models = []
all_models = list(set(async_models + local_models))

# Initialize encoder
enc = tiktoken.get_encoding("cl100k_base")

async def count_points(answer):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Count the number of different points in the following text. Respond only with a number.\n\n{answer}"
            }],
            temperature=0,
            max_tokens=2
        )
        return int(response.choices[0].message.content.strip())
    except:
        return 0

async def process_model(model):
    base_path = Path("../../Logs") / model
    
    # Process DA_results_decision_consequence.json
    decision_path = base_path / "DA_results_decision_concequence.json"
    if decision_path.exists():
        with open(decision_path) as f:
            data = json.load(f)
            total_tokens = 0
            for item in data:
                response = item.get("Response", "").split('</think>')[-1]
                total_tokens += len(enc.encode(response))
            avg_decision = total_tokens / len(data) if data else 0
    else:
        avg_decision = 0

    # Process DA_result_scenario_hazards.json
    scenario_path = base_path / "DA_result_scenario_hazards.json"
    if scenario_path.exists():
        with open(scenario_path) as f:
            data = json.load(f)
            
            # Keep original statistics variables
            answer_tokens = {f"Answer_{i}": [] for i in range(1,5)}
            points_count = {f"Answer_{i}": [] for i in range(1,5)}
            lab_issues_stats = {
                "Most_Common_Hazards": [],
                "Improper_Operation_Issues": [],
                "Negative_Lab_Environment_Impacts": [],
                "Most_Likely_Safety_Incidents": []
            }
            
            # Add comparison statistics
            answer_key_mapping = {
                1: "Most_Common_Hazards",
                2: "Improper_Operation_Issues",
                3: "Negative_Lab_Environment_Impacts",
                4: "Most_Likely_Safety_Incidents"
            }
            below_ratio = {f"Answer_{i}": [] for i in range(1,5)}

            # Process each sample
            for item in data:
                # Keep original statistics
                for i in range(1,5):
                    answer = item.get(f"Model_Answer_{i}", "").split('</think>')[-1]
                    tokens = len(enc.encode(answer))
                    answer_tokens[f"Answer_{i}"].append(tokens)
                
                lab_issues = item.get("LabSafety_Related_Issues", {})
                # Keep lab_issues statistics
                for key in lab_issues_stats:
                    lab_issues_stats[key].append(len(lab_issues.get(key, [])))

                # Add comparison logic
                tasks = []
                for i in range(1,5):
                    answer = item.get(f"Model_Answer_{i}", "").split('</think>')[-1]
                    tasks.append(count_points(answer))
                
                points_list = await asyncio.gather(*tasks)
                
                for i in range(1,5):
                    gt_length = len(lab_issues.get(answer_key_mapping[i], []))
                    model_points = points_list[i-1]
                    below_ratio[f"Answer_{i}"].append(1 if model_points < gt_length else 0)
                    points_count[f"Answer_{i}"].append(model_points)  # Keep points statistics

            # Keep original average calculations
            avg_scenario_tokens = {
                f"Answer_{i}": sum(vals)/len(vals) if vals else 0 
                for i, vals in enumerate(answer_tokens.values(), 1)
            }
            
            avg_lab_issues = {
                key: sum(vals)/len(vals) if vals else 0 
                for key, vals in lab_issues_stats.items()
            }

            # Calculate new comparison statistics
            below_ratio_avg = {
                key: sum(values)/len(values) if values else 0
                for key, values in below_ratio.items()
            }

            # Keep points average calculation
            points_avg = {
                f"Answer_{i}": sum(points_count[f"Answer_{i}"])/len(points_count[f"Answer_{i}"]) 
                if points_count[f"Answer_{i}"] else 0 
                for i in range(1,5)
            }

            # Add overall average calculation
            scenario_avg = {
                "tokens": sum(avg_scenario_tokens.values()) / 4,
                "points": sum(points_avg.values()) / 4,
                "below_ratio": sum(below_ratio_avg.values()) / 4,
                "lab_issues": sum(avg_lab_issues.values()) / 4
            }
    else:
        # Keep original initialization
        lab_issues_stats = {
            "Most_Common_Hazards": [],
            "Improper_Operation_Issues": [],
            "Negative_Lab_Environment_Impacts": [],
            "Most_Likely_Safety_Incidents": []
        }
        avg_scenario_tokens = {f"Answer_{i}": 0 for i in range(1,5)}
        avg_lab_issues = {key: 0 for key in lab_issues_stats}
        points_avg = {f"Answer_{i}": 0 for i in range(1,5)}
        below_ratio_avg = {f"Answer_{i}": 0 for i in range(1,5)}
        scenario_avg = {
            "tokens": 0,
            "points": 0,
            "below_ratio": 0,
            "lab_issues": 0
        }

    return {
        "model": model,
        "decision_avg_tokens": avg_decision,
        "scenario_avg_tokens": avg_scenario_tokens,
        "points_avg": points_avg,
        "lab_issues_avg": avg_lab_issues,
        "below_labissues_ratio": below_ratio_avg,
        "scenario_overall_avg": scenario_avg  # Add overall average
    }

async def main():
    for model in all_models:
        try:
            print(f"\n{'='*30}")
            print(f"Starting to process {model}...")
            start_time = time.time()
            
            # Process single model
            result = await process_model(model)
            
            # Output results immediately
            print(f"\nProcessing completed, time used: {time.time()-start_time:.1f} seconds")
            print(f"Model: {result['model']}")
            print(f"Decision average token count: {result['decision_avg_tokens']:.1f}")
            print("Scenario analysis:")
            print(f"  Average token count - {result['scenario_avg_tokens']}")
            print(f"  Average points count - {result['points_avg']}")
            print(f"  Below safety standard ratio - {result['below_labissues_ratio']}")
            print(f"  Lab issues average count - {result['lab_issues_avg']}")
            # Output overall average values
            print(f"  Scenario comprehensive average:")
            print(f"    Token count: {result['scenario_overall_avg']['tokens']:.1f}")
            print(f"    Points count: {result['scenario_overall_avg']['points']:.1f}")
            print(f"    Below standard ratio: {result['scenario_overall_avg']['below_ratio']:.2f}")
            print(f"    Lab issues count: {result['scenario_overall_avg']['lab_issues']:.1f}")
            print(f"{'='*30}")
            
            # Save to file immediately
            with open("analysis_results.json", "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"\n{'!'*30}")
            print(f"Error processing {model}: {str(e)}")
            print(traceback.format_exc())
            print(f"{'!'*30}")
            continue

    print("\nAll models processed!")

if __name__ == "__main__":
    asyncio.run(main()) 