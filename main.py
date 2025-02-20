from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import pandas as pd
from tqdm import tqdm


llm_cot_prompt_llama31 = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nFor the given query including a meal description, think step by step as follows:\n1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn\'t relate to the item name and serving size.\n2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for the specific serving size.\n3. Respond with a dictionary object containing the total carbohydrates in grams as follows:\n{{"total_carbohydrates": total grams of carbohydrates for the serving}}\nFor the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don\'t know the answer, set the value of "total_carbohydrates" to -1.\n\nFollow the format of the following examples when answering\n\nQuery: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."\nAnswer: Let\'s think step by step.\nThe meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n1 cup of oatmeal has 27g carbs.\n1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.\n1 glass of orange juice has 26g carbs.\nSo the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5\nOutput: {{"total_carbohydrates": 66.5}}\n\nQuery: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."\nAnswer: Let\'s think step by step.\nThe meal consists of scrambled eggs made with 2 eggs and 1 toast.\nScrambled eggs made with 2 eggs has 2g carbs.\n1 toast has 13g carbs.\nSo the total grams of carbs in the meal = (2 + 13) = 15\nOutput: {{"total_carbohydrates": 15}}\n\nQuery: "Half a peanut butter and jelly sandwich."\nAnswer: Let\'s think step by step.\nThe meal consists of 1/2 a peanut butter and jelly sandwich.\n1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (25.3*(1/2)) = 25.3g carbs\nSo the total grams of carbs in the meal = 25.3\nOutput: {{"total_carbohydrates": 25.3}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuery: "{query}"\nAnswer: Let\'s think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
llm_cot_prompt_gemma2 = '<bos><start_of_turn>user\nFor the given query including a meal description, think step by step as follows:\n1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn\'t relate to the item name and serving size.\n2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for the specific serving size.\n3. Respond with a dictionary object containing the total carbohydrates in grams as follows:\n{{"total_carbohydrates": total grams of carbohydrates for the serving}}\nFor the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don\'t know the answer, set the value of "total_carbohydrates" to -1.\n\nFollow the format of the following examples when answering\n\nQuery: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."\nAnswer: Let\'s think step by step.\nThe meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n1 cup of oatmeal has 27g carbs.\n1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.\n1 glass of orange juice has 26g carbs.\nSo the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5\nOutput: {{"total_carbohydrates": 66.5}}\n\nQuery: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."\nAnswer: Let\'s think step by step.\nThe meal consists of scrambled eggs made with 2 eggs and 1 toast.\nScrambled eggs made with 2 eggs has 2g carbs.\n1 toast has 13g carbs.\nSo the total grams of carbs in the meal = (2 + 13) = 15\nOutput: {{"total_carbohydrates": 15}}\n\nQuery: "Half a peanut butter and jelly sandwich."\nAnswer: Let\'s think step by step.\nThe meal consists of 1/2 a peanut butter and jelly sandwich.\n1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (25.3*(1/2)) = 25.3g carbs\nSo the total grams of carbs in the meal = 25.3\nOutput: {{"total_carbohydrates": 25.3}}\n\nQuery: {query}\nAnswer: Let\'s think step by step.<end_of_turn>\n<start_of_turn>model\n'


def get_response(model, tokenizer, query):
    input_text = llm_cot_prompt_gemma2.format(query=query)
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **input_ids, 
        max_new_tokens=4096,
        do_sample=False,
        temperature=0.0,
    )
    return tokenizer.decode(outputs[0])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def clean_output(raw_output, query, method_name, nutrition_name):
    prompt = llm_cot_prompt_gemma2
    remove_str = prompt.format(query=query)
    raw_output = raw_output.replace(remove_str, "")

    if "cot" in method_name:
        # discard all output which is part of the reasoning process
        splits = raw_output.split("Output:")
        if len(splits) > 1: # split into reasoning and answer part
            raw_output = splits[1]
                
    raw_output = raw_output.strip()
    # print(f"Raw output: {raw_output}")
    
    # match this pattern to find the total carb estimate
    if nutrition_name == 'fat':
        pattern = r'["\']\s*total_fat["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'protein':
        pattern = r'["\']\s*total_protein["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'energy':
        pattern = r'["\']\s*total_energy["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'carb':
        pattern = r'["\']\s*total_carbohydrates["\']:\s*(?:["\']?(-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)["\']?|\[(-?[0-9]+(?:\.[0-9]*)?(?:,\s*-?[0-9]+(?:\.[0-9]*)?)*)\])'

    else:
        raise NotImplementedError
    
    match = re.search(pattern, raw_output)
    if match:
        if match.group(1):
            pred_carbs = match.group(1) # extract the numeric part
            if is_number(pred_carbs):
                return float(pred_carbs)
            else:
                # check if output is a range
                pred_carbs_list = pred_carbs.split('-')
                if len(pred_carbs_list) == 2 and is_number(pred_carbs_list[0]) and is_number(pred_carbs_list[1]):
                    p0 = float(pred_carbs_list[0])
                    p1 = float(pred_carbs_list[1])
                    return (p0+p1)/2.0
                else:
                    print(f"EXCEPTION AFTER MATCHING")
                    print(f"Matched output: {raw_output}")
                    print(f"Query: {query}")
                    return -1
        elif match.group(2):
            try:
                pred_carbs_list = match.group(2).split(',')
                p0 = float(pred_carbs_list[0])
                p1 = float(pred_carbs_list[1])
                return (p0+p1)/2.0
            except:
                print(f"EXCEPTION AFTER MATCHING")
                print(f"Matched output: {raw_output}")
                print(f"Query: {query}")
                return -1
    else:
        if is_number(raw_output):
            return float(raw_output)
        else:
            print(f"EXCEPTION")
            print(f"Matched output: {raw_output}")
            print(f"Query: {query}")
            return -1
        

if __name__ == "__main__":
    model_path = "gemma-2-27b-it-FT-lora-ep2-cleaned10-cot/checkpoint-3510"
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    # model = PeftModel.from_pretrained(model, adapter_name)

    query = "A large size of latte"
    response = get_response(model, tokenizer, query)
    print(response)

    '''
    For test nutribench

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("dongx1997/NutriBench")["who_meal_natural"]
    # ds = ds.select(range(10))
    queries = ds['meal_description']

    responses = []
    preds = []
    for query in tqdm(queries):
        response = get_response(model, tokenizer, query)
        responses.append(response)
        pred = clean_output(response, query, "cot", "carb")
        preds.append(pred)

        
    # compare with previous results
    df_prev = pd.read_csv("cot_llm_gemma-2-27b-it-FT-lora-ep2-cleaned10-cot_who_meal_natural_query_processed_carb_1.csv")
    preds_prev = []
    for query in queries:
        pred = df_prev[df_prev['query_processed'] == query]['pred_carb'].values[0]
        preds_prev.append(pred)

    df = {
        "meal_description": queries,
        "preds": preds,
        "preds_prev": preds_prev,
        "gts": ds['carb'],
    }
    df = pd.DataFrame(df)
    df.to_csv("gemma2_carb_preds_full.csv", index=False)

    '''
