
# NutriBench

### Welcome to the official repository for NutriBench.
_A Dataset for Evaluating Large Language Models on Nutrition Estimation from Meal Descriptions_

[🌐 Project Page](https://mehak126.github.io/nutribench.html) | [📝 Paper (ICLR 2025)](https://arxiv.org/abs/2407.12843) | [📊 Dataset](https://huggingface.co/datasets/dongx1997/NutriBench) | [🔗 Github](https://github.com/DongXzz/NutriBench)

---

## News

- [2025/04/08] **NutriBench v2** is released! Now supports **24 countries** with improved **diversity** in meal descriptions.

- [2025/03/16] We’ve launched LLM-Based Carb Estimation via Text Message!  
  - For US phone numbers, text your meal description to **+1 (866) 698-9328**.  
  - For WhatsApp, send a message to **+1 (555) 730-0221**.
  
- [2025/02/11] 🎉 Our **NutriBench** paper has been **accepted at ICLR 2025**!

- [2024/10/16] Released **NutriBench v1**, the **First** benchmark for evaluating nutrition estimation from meal descriptions.  

---

## Dataset

Please refer to our [🔗 Dataset](https://huggingface.co/datasets/dongx1997/NutriBench)

---

## Inference

```bash
python inference.py
```

This script will take meal descriptions as input and return estimated carbohydrate values using a pretrained LLM.


---

## Benchmark

We currently use [**lm-evaluation-harness**](https://github.com/EleutherAI/lm-evaluation-harness) to benchmark models on NutriBench.

> 🛠️ We are working on merging our NutriBench task into the main repo via a pull request.

To benchmark your model on NutriBench **before the merge**, follow these steps:

1. Clone the `lm-evaluation-harness` repository:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

2. Copy the NutriBench task folder into `lm_eval/tasks`:

```bash
cp -r [path_to_nutribench_repo]/nutribench ./lm_eval/tasks/
```

3. Run the benchmark command (example for vLLM):

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=[model_path],tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
  --batch_size auto \
  --tasks nutribench_v2_cot \
  --output_path results \
  --seed 42 \
  --log_samples \
  --apply_chat_template
```

You can change `nutribench_v2_cot` to other tasks (e.g., `nutribench_v2_base`, etc.) depending on your use case. Please refer to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for detailed documentation.


### Reference Result

| Task                | Accuracy@7.5 | MAE     |
|---------------------|--------------|-------------|
| nutribench_v2_base  | 0.3301       | 36.18       |
| nutribench_v2_cot  | 0.3527       | 37.17       |

> These reference results were obtained using Meta-Llama-3.1-8B-Instruct.

---

## Citation

If you find **NutriBench** helpful, please consider citing:

```bibtex
@article{hua2024nutribench,
  title={NutriBench: A Dataset for Evaluating Large Language Models on Nutrition Estimation from Meal Descriptions},
  author={Hua, Andong and Dhaliwal, Mehak Preet and Burke, Ryan and Pullela, Laya and Qin, Yao},
  journal={arXiv preprint arXiv:2407.12843},
  year={2024}
}
```

---
