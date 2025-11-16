# Outcome & Process Reward Models (MVP Demos)

This repo provides minimal, runnable implementations of:

- **Outcome Reward Model (ORM)** – predicts whether a full solution is correct.
- **Process Reward Model (PRM)** – scores intermediate reasoning steps.

Both are designed to closely match the definitions and notation in the Reward Modeling chapter of Dr. Nathan Lambert’s RLHF book, and each script comes with:

1. A small training pipeline (LoRA + quantization for efficiency).
2. A toy evaluation example.
3. A book-style minimal code snippet (below).

## Quickstart: Run in Colab

You can run the full ORM and PRM MVP pipelines directly in Colab on a GPU T4 (no local setup):

- **ORM Colab:** [Open in Colab](https://colab.research.google.com/drive/1LKS9Fw_nhKQVlvxJWWOjT1MuX5j8Zvy4?usp=sharing)  
- **PRM Colab:** [Open in Colab](https://colab.research.google.com/drive/15ZFb7_YJzud6qR97h_2ZUemO3hwK6WAL?usp=sharing)

---

## Files

- `reward_orm_mvp.py`  
  - Fine-tunes **Qwen/Qwen3-1.7B-Base** as an **Outcome Reward Model**.
  - Uses a small subset of **GSM8K**:
    - For each math problem, creates a **correct** solution and a **corrupted** solution (wrong final answer).
    - Labels prompt tokens with `-100` and completion tokens with a **single outcome label**: `1` (correct) or `0` (incorrect), repeated across the completion.
  - Trains a **per-token correctness classifier** with BCE and then scores a fresh correct/incorrect pair by averaging per-token probabilities over the completion.

- `reward_prm_mvp.py`  
  - Fine-tunes **Qwen/Qwen3-0.6B-Base** as a **Process Reward Model**.
  - Uses **PRM800K**:
    - Extracts problems and **step-level ratings** in `{−1, 0, 1}`.
    - Appends a step separator after each reasoning step and labels **only the last token of each step** with the class index (mapping `{−1, 0, 1} → {0, 1, 2}`); all other tokens are `-100`.
  - Trains a **3-class step classifier** with cross-entropy and then scores an unseen reasoning trace step by step at the separator tokens.

---

## Book-Style Minimal Snippets

The snippets below are distilled from the full scripts: start from a base LM, add a small head, and define the core loss assuming a dataloader that already constructs `input_ids`, `attention_mask`, and `labels`.

### Outcome Reward Model (ORM)

**Idea:**  
Treat each labeled token in the completion as a Bernoulli trial for “this solution is correct,” with the same binary outcome label repeated across the completion. Prompt tokens are masked out.

- Prompt tokens: `labels = -100` (ignored).
- Completion tokens: `labels ∈ {0, 1}` (same label for the whole completion).
- Train with **binary cross-entropy** on the masked completion tokens.

```python
import torch.nn as nn
import torch.nn.functional as F

class OutcomeRewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One scalar logit per token: (batch, seq_len)
        logits = self.head(hidden).squeeze(-1)

        loss = None
        if labels is not None:
            # Only compute loss on completion tokens (labels 0 or 1)
            mask = labels != -100
            if mask.any():
                loss = F.binary_cross_entropy_with_logits(
                    logits[mask], labels[mask].float()
                )
            else:
                # Fully masked batch; keep graph but contribute zero
                loss = logits.sum() * 0
        return loss, logits
```

### Process Reward Model (PRM)

**Idea:**  
Assign a discrete rating to each **reasoning step** in a chain-of-thought by predicting at step boundaries only.

- Insert a **step separator** (e.g. `"\n<step>\n"`) after every step.
- For each step, set `labels = -100` on all tokens except the **final token of the step**, which holds the class index (for example mapping `{−1, 0, 1} → {0, 1, 2}`).
- Train with **cross-entropy** only on those step-boundary tokens.

```python
import torch.nn as nn
import torch.nn.functional as F

class ProcessRewardModel(nn.Module):
    def __init__(self, base_lm, num_classes=3):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One logit vector per token: (batch, seq_len, num_classes)
        logits = self.head(hidden)

        loss = None
        if labels is not None:
            # Only compute loss at step boundaries (where labels != -100)
            mask = labels != -100
            if mask.any():
                loss = F.cross_entropy(
                    logits[mask], labels[mask]
                )
            else:
                # Fully masked batch; keep graph but contribute zero
                loss = logits.sum() * 0
        return loss, logits
```

### Most Simplified (Core Loss)

These ultra-minimal snippets show just the loss computation, matching Section 7.3 style in the book.

**ORM:**
```python
# Assume model already has: model.lm (backbone) + model.head
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits_per_token = model.head(hidden).squeeze(-1)  # (batch, seq_len)

# Binary labels: 1=correct, 0=incorrect (prompt tokens masked as -100)
mask = labels != -100
loss = F.binary_cross_entropy_with_logits(
    logits_per_token[mask], labels[mask].float()
)
```

**PRM:**
```python
# Assume model outputs 3-class logits per token
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits = model.head(hidden)  # (batch, seq_len, 3)

# 3-class labels at step boundaries only: 0=-1, 1=0, 2=1 (others masked as -100)
mask = labels != -100
loss = F.cross_entropy(logits[mask], labels[mask])
```

