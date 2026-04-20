import torch
from rule_scenario_translation import generate_scenarios, multi_rule_scenario_translation
from load_model import load_model
import random
from run_smt import run_smt_code_for_sat
import os
import pandas as pd
from postprocessing import clear_memory

def verification_pipeline(model_translation_map, model_scenario_id, doc_text, num_compliant=3, num_non_compliant=3, max_tokens=1024):
    scenario_model, scenario_tokenizer = load_model(model_scenario_id)
    scenario_map = generate_scenarios(scenario_model, scenario_tokenizer, doc_text, num_compliant, num_non_compliant, max_tokens)
    del scenario_model
    del scenario_tokenizer
    clear_memory()
    scenario_arr = []

    scenario_arr.extend(scenario_map["compliant"])
    scenario_arr.extend(scenario_map["non-compliant"])

    random.shuffle(scenario_arr)

    rules_with_scenarios = []

    for scenario in scenario_arr:
        text = f"Rule Text: {doc_text}\n\n\nScenario: {scenario}"
        rules_with_scenarios.append(text)

    df = pd.DataFrame({'ModelName': [], 'ModelID': [], 'RuleScenario': [], 'SMTCode': [], 'Output': []})
    
    for model_name, model_id in model_translation_map.items():
        generated_smt_codes = multi_rule_scenario_translation(model_name, model_id, rules_with_scenarios, max_tokens)

        smt_outputs = []
        
        for idx, smt_code in enumerate(generated_smt_codes):
            smt_output = run_smt_code_for_sat(smt_code)
            df.loc[len(df)] = [model_name, model_id, rules_with_scenarios[idx], smt_code, smt_output]
        
        clear_memory()

    base_name = "verf_pipeline"
    counter = 1

    while True:
        curr_fn = f"{base_name}_{counter}.csv"
        if not os.path.exists(curr_fn):
            break
        counter += 1

    df.to_csv(curr_fn, index=False)

        