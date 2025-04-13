# 🧠 GRPO + Reward Model: LLM-based Reinforcement Optimization Framework

This project combines large language models (LLMs), the GRPO reinforcement learning algorithm, and a LoRA-based CTR-style reward model. The reward model scores generated outputs to guide the LLM training via reinforcement learning.

---

## 📌 Highlights

- 🚀 Train LLMs with GRPO for text generation optimization  
- 🧠 Train a LoRA-based Reward Model for CTR-style preference scoring  
- ⚙️ Combine multiple reward strategies (format, length, CTR scoring)  
- 💦 Modular design for easy integration into RLHF / SFT pipelines  

---

## 📁 Project Structure

```
.
├── train_reward_model/         # Reward model training and architecture
│   ├── data.py
│   └── reward_model.py     # train scorer
├── train_grpo/         # Reward model training and architecture
│   ├── ctr_scorer.py
│   └── step2.py     # train grpo
└── README.md
```

---

## 🏗️ Architecture & Workflow

### 1️⃣ Train the Reward Model (CTR-style classifier)

Using `Qwen2.5-3B-Instruct` + LoRA and a custom MLP classifier to predict user preference:

```
Qwen2.5-3B (output_hidden_states=True)
    ↓
Last Layer Avg Pooling
    ↓
MLP Classifier (1024→512→1) → Sigmoid
```

#### Training Command

```bash
python train_reward_model/reward_model.py
```

Default settings:

- Loss: `BCEWithLogitsLoss`
- Optimizer: AdamW
- Precision: `bfloat16`
- Metrics: Accuracy + AUC
- Output: `best_model.pth`

---

### 2️⃣ Use Reward Model as a Scorer

The trained reward model can be used to score generated answers:

```python
from ctr_scorer import CTRScorer

scorer = CTRScorer(
    model_path="Qwen2.5-3B-Instruct",
    ckpt_path="best_model.pth",
    device="cuda"
)

score = scorer.predict("LLM generated response")
print(f"CTR Score: {score:.4f}")
```

---

### 3️⃣ GRPO Training with Reward Integration

In `grpo_with_ctr.py`, we define five types of reward functions:

| Reward Function | Description |
|------------------|-------------|
| `strict_format_reward_func` | Enforces correct `<think>` block usage |
| `soft_format_reward_func`   | Loose format verification |
| `xmlcount_reward_func`      | Tag count scoring |
| `reasoning_length_reward_func` | Reasoning text length scoring |
| `ctr_reward_func`           | Uses CTRScorer to evaluate answers |

#### Training Command

```bash
python step2.py
```

Each generation and its reward score is logged in:

```
grpo_ctr_debug_log.jsonl
```

#### Output Directory:

```
grpo_saved_model/
```

---

## ⚙️ Requirements

- Python >= 3.8  
- torch >= 2.0  
- transformers >= 4.36  
- peft >= 0.7  
- accelerate >= 0.21  
- trl >= 0.7.10  
- datasets >= 2.14  
- scikit-learn  

Install via:

```bash
pip install -r requirements.txt
```

---

## ✨ Use Cases

- Preference-based LLM generation optimization  
- High-CTR headline generation  
- Simulated RLHF reward design and ablation  

---

## 🪪 License

This project is licensed under the MIT License. Feel free to use, modify, and contribute.

---

## 🤝 Contact

Interested in RLHF, GRPO, or LLM preference modeling? Contributions and feedback are welcome!  
Feel free to open issues or reach out directly.
