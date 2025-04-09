from transformers import AutoTokenizer, AutoModelForCausalLM


prompt = '''For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for the specific serving size.
3. Respond with a dictionary object containing the total carbohydrates in grams as follows:
{"total_carbohydrates": total grams of carbohydrates for the serving}
For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 27g carbs.
1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.
1 glass of orange juice has 26g carbs.
So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5
Output: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 2g carbs.
1 toast has 13g carbs.
So the total grams of carbs in the meal = (2 + 13) = 15
Output: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.
So the total grams of carbs in the meal = 25.3
Output: {"total_carbohydrates": 25.3}'''


def get_response(model, tokenizer, query):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f'Query: {query}\nAnswer: Let\'s think step by step.'}
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=4096,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0])

if __name__ == "__main__":
    model_path = "<PATH to Model>"
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )

    query = "One mango ice cream"
    response = get_response(model, tokenizer, query)
    print(response)
