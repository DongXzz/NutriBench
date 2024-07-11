
# NutriBench

### Welcome to the official repository for NutriBench.
_A Dataset for Evaluating Large Language Models in Carbohydrate Estimation from Meal Descriptions_

[[Website](https://mehak126.github.io/nutribench.html)]

All data can be accessed in the [`data`](data) directory.

NutriBench consists of 5,000 human-verified meal descriptions with macro-nutrient labels, including carbohydrates, proteins, fats, and calories. It is divided into 15 subsets varying in meal description complexity. 

The subsets are named according to the following format: 
`{serving_unit}_{retrieval_type}_{number_of_food}_{number_of_serving}.csv`.

Where:
* `{serving_unit}`: how the servings are measured in the description. `{natural}` (such as '1 cup') or `{metric}` (such as '50g').
* `{retrieval_type}`: to evaluate performance on RAG based methods we divided Nutribench in two. With the `{direct}` subsets, food items can be directly retrieved from a RAG DB with exact food name matches. With the `{indirect}` subsets there is no direct match between the queried food item name and the items in the RAG DB.
* `{number_of_food}`: the number of discrete food items in the description. One item `{single}`, two items `{double}`, or three items `{triple}`.
* `{number_of_serving}`: the number of servings in each description. One serving `{single}`, otherwise `{multiple}`.

For example:

`natural_direct_single_multiple.csv` indicates queries with natural serving unit, direct retrieval, a single food , and multiple servings of that single food.

### Load Data (Using pandas)
```bash
# pip install pandas
import pandas as pd

file_path = 'data/metric_indirect_double_multiple.csv'
df = pd.read_csv(file_path)
