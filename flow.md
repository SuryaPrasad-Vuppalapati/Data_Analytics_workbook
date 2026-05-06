```
Source Document
      │
      ▼
┌─────────────────────────────────┐
│  Summarization (BART / T5 / PEGASUS) │
└─────────────────────────────────┘
      │
      ▼
Generated Summary (cached, shared)
      │
      ├──────────────────────────────────────────┐
      │                                          │
      ▼                                          ▼
Pipeline 1                              Pipeline 3
Generate K more summaries               Extract token log-probs
Compare consistency across them         Compute perplexity & entropy
Score each sentence                     Train logistic classifier
      │                                          │
      │         Pipeline 2                       │
      │    NLI check: source ⊨ sentence?         │
      │    SummaC / MiniCheck / AlignScore       |
      │         │                                │
      └─────────┴────────────────────────────────┘
                         │
                         ▼
             Sentence-level scores
             (one score per pipeline per sentence)
                         │
                         ▼
          Compare against FaithBench labels
          (hallucinated / faithful / intrinsic / extrinsic)
                         │
                         ▼
        AUROC · F1 · ECE · Latency per method
                         │
                         ▼
        Final Cross-Method Comparison Table
```

# FACTSHIELD — HuggingFace Execution Plan

**Project:** Hallucination Detection in LLM-Generated Summaries  
**Team:** Surya Prasad Vuppalapati · Emmanuel Nischay Gapti · Manikanta Amareswarapu  
**Platform:** HuggingFace (datasets, models, compute, storage) + GitHub (code)  
**Cost:** $0

---

## Table of Contents

1. [Repository Tree](#1-repository-tree)
2. [HuggingFace Organisation Setup](#2-huggingface-organisation-setup)
3. [Phase 0 — Environment and Repository](#3-phase-0--environment-and-repository)
4. [Phase 1 — Data Ingestion](#4-phase-1--data-ingestion)
5. [Phase 2 — Shared Summarization Cache](#5-phase-2--shared-summarization-cache)
6. [Phase 3 — Pipeline 1: SelfCheckGPT (Surya)](#6-phase-3--pipeline-1-selfcheckgpt-surya)
7. [Phase 3 — Pipeline 2: Entailment Verification (Nischay)](#7-phase-3--pipeline-2-entailment-verification-nischay)
8. [Phase 3 — Pipeline 3: Token Probability (Manik)](#8-phase-3--pipeline-3-token-probability-manik)
9. [Phase 4 — Evaluation](#9-phase-4--evaluation)
10. [Phase 5 — Final Comparison Table](#10-phase-5--final-comparison-table)
11. [Phase 6 — Integration and Delivery](#11-phase-6--integration-and-delivery)
12. [Week-by-Week Timeline](#12-week-by-week-timeline)

---

## 1. Repository Tree

Everything below lives on GitHub. No data, no model weights, no cache files go into this repo — those all go to HuggingFace Hub.

```
factshield/
|
|-- configs/
|   |-- task1_v1_scoring_sweep.yaml
|   |-- task1_v2_k_ablation.yaml
|   |-- task1_v3_crossdomain.yaml
|   |-- task2_v1_summac.yaml
|   |-- task2_v2_minicheck.yaml
|   |-- task2_v3_alignscore.yaml
|   |-- task3_v1_perplexity.yaml
|   |-- task3_v2_entropy.yaml
|   |-- task3_v3_classifier.yaml
|   `-- template.yaml
|
|-- pipelines/
|   |-- shared/
|   |   |-- data_loader.py        # loads datasets from HF Hub
|   |   |-- splitter.py           # creates stratified splits, pushes to Hub
|   |   |-- generate.py           # summarization cache generation
|   |   `-- evaluate.py           # shared metrics: AUROC, F1, ECE, latency
|   |
|   |-- task1_selfcheck/
|   |   |-- run.py                # entry point: python run.py --config ...
|   |   |-- pipeline.py           # orchestrates V1, V2, V3
|   |   |-- scorer.py             # BERTScore, NLI, N-gram scorers
|   |   |-- evaluate.py           # task1-specific evaluation and plots
|   |   `-- README.md
|   |
|   |-- task2_entailment/
|   |   |-- run.py
|   |   |-- pipeline.py
|   |   |-- scorer.py             # SummaC, MiniCheck, AlignScore
|   |   |-- evaluate.py
|   |   `-- README.md
|   |
|   `-- task3_token_prob/
|       |-- run.py
|       |-- pipeline.py
|       |-- scorer.py             # perplexity, entropy extraction
|       |-- classifier.py         # logistic regression + Platt scaling
|       |-- evaluate.py
|       `-- README.md
|
|-- tests/
|   |-- test_data.py
|   |-- test_evaluate.py
|   |-- test_task1.py
|   |-- test_task2.py
|   `-- test_task3.py
|
|-- scripts/
|   |-- assemble_final_table.py   # pulls all 3 task CSVs from Hub, builds final table
|   `-- plot_style.mplstyle       # shared matplotlib style for all plots
|
|-- docs/
|   |-- final_report.md
|   `-- experimental_notes.md
|
|-- .github/
|   `-- workflows/
|       `-- ci.yml                # runs pytest on every PR to main
|
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## 2. HuggingFace Organisation Setup

**Owner: Manik. Do this before writing any pipeline code.**

This is a one-time setup that takes about 30 minutes. Once done, all three members share the same workspace.

### 2.1 Create the Organisation

Go to `huggingface.co/organizations/new` and create an organisation named `factshield-team`. Then invite Surya and Nischay from the Settings > Members page with Write access.

### 2.2 Create Four Private Repositories

Create these four repos inside the organisation. Each serves a specific purpose and replaces what you would otherwise store on Google Drive.

| Repository | Type | Purpose |
|---|---|---|
| `factshield-team/cache` | Dataset | Stores all generated summaries, K-samples, and token scores |
| `factshield-team/results` | Dataset | Stores task1, task2, task3 score CSVs and the final table |
| `factshield-team/faithbench-labels` | Dataset | Stores Nischay's typed annotation labels |
| `factshield-team/models` | Model | Stores the trained Task 3 logistic classifier |

To create each: click your org → New Dataset (or New Model) → set name → set to Private → Create.

### 2.3 Get Your HuggingFace Token

Each member goes to `huggingface.co/settings/tokens`, creates a token with Write access, and saves it. Add the token to your `.env` file as `HF_TOKEN=hf_xxxxx`. Never commit this file.

Authenticate at the top of every script:

```python
from huggingface_hub import login
login(token='hf_YOUR_TOKEN')
```

### 2.4 Create HuggingFace Spaces

Each pipeline gets a hosted Space where you write and run your code. Create these four Spaces inside the organisation:

| Space | Owner |
|---|---|
| `factshield-team/cache-generator` | Manik |
| `factshield-team/task1-selfcheck` | Surya |
| `factshield-team/task2-entailment` | Nischay |
| `factshield-team/task3-token-prob` | Manik |

For each: New Space > SDK: Gradio > Hardware: CPU Basic (free) > Create. When you need GPU for a heavy run, go to the Space settings and enable ZeroGPU — this gives you free T4 burst access.

Each Space is backed by a Git repo. Push your pipeline scripts there and they run in the cloud:

```bash
git clone https://huggingface.co/spaces/factshield-team/task1-selfcheck
cd task1-selfcheck
# copy your pipeline scripts here
git add . && git commit -m "add selfcheck pipeline"
git push
```

---

## 3. Phase 0 — Environment and Repository

**Owner: All members. Complete before writing any pipeline code.**

### 3.1 Create the GitHub Repository

Manik creates the GitHub repo `factshield`. Enable branch protection on `main` — all changes must come through a pull request with at least one reviewer. Invite Surya and Nischay as collaborators with Write access.

### 3.2 Clone and Set Up Locally

Every member runs:

```bash
git clone https://github.com/<org>/factshield.git
cd factshield
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3.3 requirements.txt

```
transformers==4.46.0
torch==2.1.0
datasets==2.20.0
huggingface_hub==0.24.0
selfcheckgpt==0.1.12
bert_score==0.3.13
sentence-transformers==2.6.0
scikit-learn==1.5.0
scipy==1.12.0
spacy==3.7.0
netcal==0.3.0
wandb
pandas
numpy
matplotlib
seaborn
joblib
nltk
```

### 3.4 .gitignore

```
.env
data/
models/
__pycache__/
*.pkl
*.h5
.ipynb_checkpoints/
*.pyc
```

### 3.5 Branching Convention

Use this naming pattern for all branches:

```
feature/task1/short-description
feature/task2/short-description
feature/task3/short-description
fix/shared/short-description
chore/short-description
```

Never commit directly to `main`. Every change goes through a PR. The CI workflow runs `pytest tests/` automatically on every PR and blocks merging if tests fail.

### 3.6 Config Template

Every experiment run is defined by a YAML config. No hardcoded hyperparameters in Python files. Create `configs/template.yaml`:

```yaml
model_name: facebook/bart-base
dataset: cnndm
variant_id: v1
K: 5
seed: 42
batch_size: 8
max_docs: 2000
wandb_run_name: task1_bart_cnndm_v1_k5
hf_cache_repo: factshield-team/cache
hf_results_repo: factshield-team/results
```

---

## 4. Phase 1 — Data Ingestion

**Owner: Nischay leads. Surya and Manik verify outputs.**  
**Week: 1**

### 4.1 Load Datasets Directly from HuggingFace Hub

Create `pipelines/shared/data_loader.py`. Datasets stream directly — nothing is downloaded manually.

```python
from datasets import load_dataset
from huggingface_hub import login

login(token='hf_YOUR_TOKEN')

# CNN/DailyMail — in-domain benchmark
cnndm = load_dataset('cnn_dailymail', '3.0.0')

# XSUM — high abstraction cross-domain test
xsum = load_dataset('EdinburghNLP/xsum')

# FaithBench — annotated hallucination benchmark
faithbench = load_dataset('vectara/FaithBench')
```

### 4.2 Create Stratified Splits

Create `pipelines/shared/splitter.py`. Lock the random seed globally before any split operation.

```python
import random
import numpy as np
from datasets import DatasetDict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def make_splits(dataset, split_key='test', n_sample=2000, seed=42):
    sampled = dataset[split_key].shuffle(seed=seed).select(range(n_sample))
    first_split = sampled.train_test_split(test_size=0.30, seed=seed)
    val_test = first_split['test'].train_test_split(test_size=0.50, seed=seed)
    return DatasetDict({
        'train': first_split['train'],   # 70% (~1400 docs)
        'val':   val_test['train'],      # 15% (~300 docs)
        'test':  val_test['test'],       # 15% (~300 docs)
    })

cnndm_splits = make_splits(cnndm)
xsum_splits  = make_splits(xsum)

# FaithBench: all 500 pairs, same 70/15/15 ratio
fb_split   = faithbench['train'].train_test_split(test_size=0.30, seed=SEED)
fb_vt      = fb_split['test'].train_test_split(test_size=0.50, seed=SEED)
fb_splits  = DatasetDict({'train': fb_split['train'],
                           'val':   fb_vt['train'],
                           'test':  fb_vt['test']})
```

### 4.3 Push Splits to HuggingFace Hub

This is the key step — everyone on the team loads from here instead of from local files or Google Drive.

```python
cnndm_splits.push_to_hub(
    'factshield-team/cache',
    config_name='cnndm_splits',
    private=True
)

xsum_splits.push_to_hub(
    'factshield-team/cache',
    config_name='xsum_splits',
    private=True
)

fb_splits.push_to_hub(
    'factshield-team/cache',
    config_name='faithbench_splits',
    private=True
)
```

Any team member loads a split with:

```python
splits = load_dataset('factshield-team/cache', 'cnndm_splits')
train_data = splits['train']
```

### 4.4 Create and Push FaithBench Typed Labels

Nischay creates the ground-truth annotation file. This file is used by all three pipelines to compute F1 and AUROC against real hallucination labels.

```python
import pandas as pd
from datasets import Dataset

# Build from FaithBench annotation fields
# Required columns: doc_id, sent_id, label_binary,
#                   label_type (intrinsic / extrinsic),
#                   label_subtype (entity / relation)
typed_labels = pd.DataFrame({
    'doc_id':        [...],
    'sent_id':       [...],
    'label_binary':  [...],   # 1 = hallucinated, 0 = faithful
    'label_type':    [...],   # 'intrinsic' or 'extrinsic'
    'label_subtype': [...],   # 'entity' or 'relation'
})

Dataset.from_pandas(typed_labels).push_to_hub(
    'factshield-team/faithbench-labels',
    private=True
)
```

Validate before pushing:

```python
assert typed_labels['doc_id'].notna().all(),        "null doc_ids found"
assert typed_labels['label_binary'].isin([0,1]).all(), "invalid binary labels"
assert set(typed_labels['label_type']) == {'intrinsic','extrinsic'}, "unexpected label types"
assert len(typed_labels) == 500, "FaithBench should have 500 annotated pairs"
print("Validation passed.")
```

### 4.5 Data Unit Tests

Create `tests/test_data.py`:

```python
from datasets import load_dataset

def test_split_sizes():
    splits = load_dataset('factshield-team/cache', 'cnndm_splits')
    total = len(splits['train']) + len(splits['val']) + len(splits['test'])
    assert abs(len(splits['train']) / total - 0.70) < 0.02

def test_faithbench_labels():
    labels = load_dataset('factshield-team/faithbench-labels')['train']
    df = labels.to_pandas()
    assert len(df) == 500
    assert df['label_binary'].notna().all()
    assert set(df['label_type']).issubset({'intrinsic','extrinsic'})

def test_seed_reproducibility():
    # Run split twice, assert identical doc_ids in train split
    from pipelines.shared.splitter import make_splits
    from datasets import load_dataset
    cnndm = load_dataset('cnn_dailymail', '3.0.0')
    s1 = make_splits(cnndm, seed=42)
    s2 = make_splits(cnndm, seed=42)
    assert list(s1['train']['id']) == list(s2['train']['id'])
```

Run: `pytest tests/test_data.py -v` — all tests must pass before Phase 2 begins.

---

## 5. Phase 2 — Shared Summarization Cache

**Owner: Manik leads. This is the most critical infrastructure step.**  
**Weeks: 1-2**

All three pipelines consume the same generated summaries. Manik generates everything once and pushes it to the Hub. Surya and Nischay load from there — they never run a summarization model themselves.

### 5.1 Load Models from HuggingFace Hub

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = {
    'bart':    'facebook/bart-base',
    't5':      'google/t5-small',
    'pegasus': 'google/pegasus-large',
}

def load_model(name):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODELS[name],
        torch_dtype=torch.float16,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODELS[name])
    return model, tokenizer
```

### 5.2 Shared Decoding Configuration

Define once in a config YAML and never hardcode it anywhere:

```yaml
# configs/decoding.yaml
beam_size: 4
max_new_tokens: 128
no_repeat_ngram_size: 3
temperature: 1.0
```

### 5.3 Generate Single Summaries (Used by Pipelines 2 and 3)

```python
from datasets import load_dataset, Dataset

def generate_summaries(model, tokenizer, dataset, model_name, dataset_name):
    results = []
    for i, sample in enumerate(dataset):
        inputs = tokenizer(
            sample['article'], return_tensors='pt',
            truncation=True, max_length=512
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                num_beams=4, max_new_tokens=128,
                no_repeat_ngram_size=3
            )
        summary = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({'doc_id': sample['id'],
                        'source_text': sample['article'],
                        'summary': summary})

        # Checkpoint every 100 docs so a dropped session loses nothing
        if i % 100 == 0 and i > 0:
            Dataset.from_list(results).push_to_hub(
                'factshield-team/cache',
                config_name=f'{model_name}_{dataset_name}_summaries',
                private=True
            )

    # Final push
    Dataset.from_list(results).push_to_hub(
        'factshield-team/cache',
        config_name=f'{model_name}_{dataset_name}_summaries',
        private=True
    )
```

### 5.4 Generate K Samples (Used by Pipeline 1 only)

```python
def generate_k_samples(model, tokenizer, dataset, model_name, dataset_name, K=10):
    results = []
    for i, sample in enumerate(dataset):
        inputs = tokenizer(
            sample['article'], return_tensors='pt',
            truncation=True, max_length=512
        )
        samples = []
        for _ in range(K):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=True, temperature=1.0,
                    max_new_tokens=128
                )
            samples.append(tokenizer.decode(out[0], skip_special_tokens=True))

        row = {'doc_id': sample['id']}
        for k, s in enumerate(samples):
            row[f'summary_k{k+1}'] = s
        results.append(row)

        if i % 100 == 0 and i > 0:
            Dataset.from_list(results).push_to_hub(
                'factshield-team/cache',
                config_name=f'{model_name}_{dataset_name}_k_samples',
                private=True
            )

    Dataset.from_list(results).push_to_hub(
        'factshield-team/cache',
        config_name=f'{model_name}_{dataset_name}_k_samples',
        private=True
    )
```

### 5.5 Generate Token Log-Probabilities (Used by Pipeline 3 only)

```python
import numpy as np

def generate_with_scores(model, tokenizer, dataset, model_name, dataset_name):
    results = []
    for i, sample in enumerate(dataset):
        inputs = tokenizer(
            sample['article'], return_tensors='pt',
            truncation=True, max_length=512
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=4, max_new_tokens=128
            )
        scores = torch.stack(out.scores, dim=1)        # (1, seq_len, vocab_size)
        log_probs = scores.log_softmax(-1)             # log probability distribution
        token_ids = out.sequences[0, 1:]               # generated token ids
        chosen_lp = log_probs[0, range(len(token_ids)), token_ids]  # per-token log-prob

        results.append({
            'doc_id':        sample['id'],
            'token_logprobs': chosen_lp.cpu().numpy().tobytes(),
            'seq_len':        len(token_ids),
        })

        if i % 100 == 0 and i > 0:
            Dataset.from_list(results).push_to_hub(
                'factshield-team/cache',
                config_name=f'{model_name}_{dataset_name}_token_scores',
                private=True
            )

    Dataset.from_list(results).push_to_hub(
        'factshield-team/cache',
        config_name=f'{model_name}_{dataset_name}_token_scores',
        private=True
    )
```

### 5.6 What Gets Generated and Pushed

Run the above functions for every combination below:

| Model | Dataset | Config Name Suffix |
|---|---|---|
| bart | cnndm | `_summaries`, `_k_samples`, `_token_scores` |
| bart | xsum | `_summaries`, `_k_samples`, `_token_scores` |
| bart | faithbench | `_summaries`, `_k_samples`, `_token_scores` |
| t5 | cnndm | same three |
| t5 | xsum | same three |
| t5 | faithbench | same three |
| pegasus | cnndm | same three |
| pegasus | xsum | same three |
| pegasus | faithbench | same three |

That is 27 configs total in `factshield-team/cache`. Once they are all pushed, Surya and Nischay never need to run a summarization model.

---

## 6. Phase 3 — Pipeline 1: SelfCheckGPT (Surya)

**Owner: Surya**  
**Weeks: 2-4**  
**Space: `factshield-team/task1-selfcheck`**

The idea: generate multiple summaries of the same document, then check how consistent each sentence is across all samples. High inconsistency means the model was unsure — likely hallucinating.

### 6.1 Load Cache from Hub

```python
from datasets import load_dataset
from huggingface_hub import login

login(token='hf_YOUR_TOKEN')

# Load K-samples generated by Manik
k_cache = load_dataset(
    'factshield-team/cache',
    'bart_cnndm_k_samples'
)['train']
k_df = k_cache.to_pandas()

# Load ground-truth labels from Nischay
labels = load_dataset(
    'factshield-team/faithbench-labels'
)['train'].to_pandas()
```

### 6.2 V1 — Three Scoring Modes at K=5

```python
from selfcheckgpt.modeling_selfcheck import (
    SelfCheckBERTScore,
    SelfCheckNLI,
    SelfCheckNgram
)

scorer_bert  = SelfCheckBERTScore(rescale_with_baseline=True)
scorer_nli   = SelfCheckNLI(nli_model='cross-encoder/nli-deberta-v3-large')
scorer_ngram = SelfCheckNgram(n=1)

results = []
for _, row in k_df.iterrows():
    # Use K=5 samples: columns summary_k1 to summary_k5
    samples  = [row[f'summary_k{k}'] for k in range(1, 6)]
    primary  = samples[0]
    others   = samples[1:]
    sentences = primary.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    bert_scores  = scorer_bert.predict(sentences=sentences,
                                        sampled_passages=others)
    nli_scores   = scorer_nli.predict(sentences=sentences,
                                       sampled_passages=others)
    ngram_scores = scorer_ngram.predict(sentences=sentences,
                                         sampled_passages=others)

    for i, sent in enumerate(sentences):
        for mode, score in [('bert', bert_scores[i]),
                             ('nli',  nli_scores[i]),
                             ('ngram',ngram_scores[i])]:
            results.append({
                'doc_id':     row['doc_id'],
                'sent_id':    i,
                'summarizer': 'bart',
                'dataset':    'cnndm',
                'K':          5,
                'score_mode': mode,
                'score':      score,
            })
```

Select best score mode by val-set AUROC. Use it for V2 and V3.

### 6.3 V2 — K Ablation (K = 3, 5, 7, 10)

Reuse the cached K=10 samples. For K=3, take `summary_k1` to `summary_k3`. For K=5, take `summary_k1` to `summary_k5`. No re-generation needed.

For each K value, record AUROC and wall-clock latency (ms per document). Plot K vs. AUROC and K vs. latency on the same dual-axis chart per summarizer. Identify the minimum K where AUROC is within 2% of K=10 and latency overhead is under 15%.

### 6.4 V3 — Cross-Domain Transfer

Apply the best (K, score mode) from V2 zero-shot on XSUM and FaithBench. Load those caches from Hub the same way. Compute delta-AUROC = in-domain AUROC minus cross-domain AUROC. Join with `faithbench-labels` to split false negatives by hallucination type (intrinsic vs. extrinsic).

### 6.5 Push Results to Hub

```python
from datasets import Dataset

Dataset.from_list(results).push_to_hub(
    'factshield-team/results',
    config_name='task1_scores',
    private=True
)
```

---

## 7. Phase 3 — Pipeline 2: Entailment Verification (Nischay)

**Owner: Nischay**  
**Weeks: 2-4**  
**Space: `factshield-team/task2-entailment`**

The idea: treat each summary sentence as a claim, treat the source document as the premise, and ask an NLI model whether the source supports the claim. If it does not, the sentence is likely hallucinated.

### 7.1 Load Cache from Hub

```python
from datasets import load_dataset
from huggingface_hub import login

login(token='hf_YOUR_TOKEN')

# Load single summaries generated by Manik
cache = load_dataset(
    'factshield-team/cache',
    'bart_cnndm_summaries'
)['train']
df = cache.to_pandas()

labels = load_dataset(
    'factshield-team/faithbench-labels'
)['train'].to_pandas()
```

### 7.2 V1 — SummaC (Zero-Shot and Conv)

```python
from summac.model_summac import SummaCZS, SummaCConv

model_zs   = SummaCZS(granularity='sentence', model_name='vitc')
model_conv = SummaCConv(models=['vitc'], bins='percentile',
                        granularity='sentence')

results = []
for _, row in df.iterrows():
    sentences = [s.strip() for s in row['summary'].split('.') if s.strip()]
    source    = row['source_text']

    for i, sent in enumerate(sentences):
        score_zs   = model_zs.score([source], [sent])['scores'][0]
        score_conv = model_conv.score([source], [sent])['scores'][0]

        results.append({'doc_id': row['doc_id'], 'sent_id': i,
                         'summarizer': 'bart', 'dataset': 'cnndm',
                         'method': 'summac_zs',   'score': score_zs})
        results.append({'doc_id': row['doc_id'], 'sent_id': i,
                         'summarizer': 'bart', 'dataset': 'cnndm',
                         'method': 'summac_conv', 'score': score_conv})
```

### 7.3 V2 — MiniCheck

```python
from minicheck.minicheck import MiniCheck

# Use roberta-large (small variant). If VRAM is tight, use roberta-base.
scorer_mini = MiniCheck(model_name='roberta-large')

for _, row in df.iterrows():
    sentences = [s.strip() for s in row['summary'].split('.') if s.strip()]
    for i, sent in enumerate(sentences):
        pred_label, score, _, _ = scorer_mini.score(
            docs=[row['source_text']],
            claims=[sent]
        )
        results.append({'doc_id': row['doc_id'], 'sent_id': i,
                         'summarizer': 'bart', 'dataset': 'cnndm',
                         'method': 'minicheck', 'score': float(score[0])})
```

### 7.4 V3 — AlignScore and Ensemble

```python
from alignscore import AlignScore

scorer_align = AlignScore(
    model='roberta-large',
    batch_size=32,
    device='cuda',
    ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt'
)

for _, row in df.iterrows():
    sentences = [s.strip() for s in row['summary'].split('.') if s.strip()]
    for i, sent in enumerate(sentences):
        align_score = scorer_align.score(
            contexts=[row['source_text']],
            claims=[sent]
        )[0]
        results.append({'doc_id': row['doc_id'], 'sent_id': i,
                         'summarizer': 'bart', 'dataset': 'cnndm',
                         'method': 'alignscore', 'score': align_score})

# Ensemble: average AlignScore and SummaC-Conv
# Join the two score columns on (doc_id, sent_id) and average them
```

### 7.5 V4 — Hallucination Type Disaggregation

Join results with `faithbench-labels` on `(doc_id, sent_id)`. Compute F1 separately for `label_type == 'intrinsic'` and `label_type == 'extrinsic'`. For entity-level analysis, use spaCy to tag NER spans and compute recall per entity category (PERSON, ORG, DATE, CARDINAL).

### 7.6 Push Results to Hub

```python
from datasets import Dataset

Dataset.from_list(results).push_to_hub(
    'factshield-team/results',
    config_name='task2_scores',
    private=True
)
```

---

## 8. Phase 3 — Pipeline 3: Token Probability (Manik)

**Owner: Manik**  
**Weeks: 2-4**  
**Space: `factshield-team/task3-token-prob`**

The idea: the summarization model itself signals uncertainty through token probabilities. High perplexity and high entropy at a token position mean the model was not confident — which correlates with hallucination. No external model needed. The signal is already there in the generation output.

### 8.1 Load Token Scores from Hub

```python
from datasets import load_dataset
import numpy as np

scores_ds = load_dataset(
    'factshield-team/cache',
    'bart_cnndm_token_scores'
)['train']
scores_df = scores_ds.to_pandas()

labels = load_dataset(
    'factshield-team/faithbench-labels'
)['train'].to_pandas()
```

### 8.2 V1 — Sentence-Level Perplexity

```python
import math

results = []
for _, row in scores_df.iterrows():
    # Deserialise bytes back to numpy array
    log_probs = np.frombuffer(row['token_logprobs'], dtype=np.float32)

    # Sentence perplexity: exp of negative mean log-prob
    # Higher perplexity = model less confident = more likely hallucinated
    perplexity = math.exp(-np.mean(log_probs))

    results.append({
        'doc_id':     row['doc_id'],
        'summarizer': 'bart',
        'dataset':    'cnndm',
        'perplexity': perplexity,
    })
```

### 8.3 V2 — Mean and Max Entropy

Entropy is computed from the full vocabulary distribution at each token position. This requires the full softmax output, which was stored separately during cache generation.

```python
# Per-token entropy: H(t) = -sum(p_i * log(p_i + 1e-9)) over all vocab items
# mean_entropy = average H(t) over all tokens in the sentence
# max_entropy  = peak H(t) within the sentence (hallucination span proxy)

def compute_entropy(full_distribution):
    # full_distribution: (seq_len, vocab_size) numpy array
    probs = np.exp(full_distribution)  # convert log-probs to probs
    entropy_per_token = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
    return float(np.mean(entropy_per_token)), float(np.max(entropy_per_token))

for _, row in scores_df.iterrows():
    mean_ent, max_ent = compute_entropy(row['full_distribution'])
    # Add to results alongside perplexity
```

### 8.4 V3 — Logistic Regression Classifier

```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Feature vector per sentence
feature_cols = ['perplexity', 'mean_entropy', 'max_entropy',
                'tail_mean_nll', 'sent_length']

X_train = results_df[feature_cols].values
y_train = results_df['label_binary'].values

# C sweep with 5-fold cross-validation
best_C, best_auroc = 1.0, 0
for C in [0.01, 0.1, 1.0, 10.0]:
    clf = LogisticRegression(C=C, penalty='l2', max_iter=1000)
    cv_scores = cross_val_score(clf, X_train, y_train,
                                 cv=StratifiedKFold(5),
                                 scoring='roc_auc')
    if cv_scores.mean() > best_auroc:
        best_auroc = cv_scores.mean()
        best_C = C

# Train final model and calibrate
final_clf = LogisticRegression(C=best_C, penalty='l2', max_iter=1000)
final_clf.fit(X_train, y_train)

calibrated = CalibratedClassifierCV(final_clf, method='sigmoid', cv='prefit')
calibrated.fit(X_val, y_val)
```

Push the trained model to Hub:

```python
import joblib
from huggingface_hub import HfApi

joblib.dump(calibrated, 'task3_classifier.pkl')

api = HfApi()
api.upload_file(
    path_or_fileobj='task3_classifier.pkl',
    path_in_repo='task3_classifier.pkl',
    repo_id='factshield-team/models',
    repo_type='model',
    token='hf_YOUR_TOKEN'
)
```

Load it from anywhere:

```python
from huggingface_hub import hf_hub_download
import joblib

path = hf_hub_download(repo_id='factshield-team/models',
                        filename='task3_classifier.pkl')
clf = joblib.load(path)
```

### 8.5 V4 — Entity-Token Entropy Profiling

Apply spaCy NER to PEGASUS-large summaries. For each NER span, retrieve the entropy values at those token positions. Compute mean entropy per entity category (PERSON, ORG, DATE, CARDINAL). Cross-reference with Nischay's entity recall results from Pipeline 2.

### 8.6 Push Results to Hub

```python
from datasets import Dataset

Dataset.from_list(results).push_to_hub(
    'factshield-team/results',
    config_name='task3_scores',
    private=True
)
```

---

## 9. Phase 4 — Evaluation

**Owner: Each member evaluates their own task.**  
**Weeks: 5-6**

### 9.1 Shared Evaluation Module

Create `pipelines/shared/evaluate.py`. All three tasks import from here — no duplicate implementations.

```python
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from netcal.metrics import ECE

def compute_auroc(scores, labels):
    return roc_auc_score(labels, scores)

def compute_auprc(scores, labels):
    return average_precision_score(labels, scores)

def compute_f1_at_best_threshold(val_scores, val_labels,
                                  test_scores, test_labels):
    # Optimize threshold on val set, never on test
    thresholds = sorted(set(val_scores))
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in val_scores]
        f = f1_score(val_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    test_preds = [1 if s >= best_t else 0 for s in test_scores]
    return f1_score(test_labels, test_preds, zero_division=0), best_t

def compute_ece(probs, labels, n_bins=10):
    ece = ECE(n_bins)
    return ece.measure(probs, labels)

def compute_latency(start_time, end_time, n_queries):
    return (end_time - start_time) * 1000 / n_queries  # ms per query
```

### 9.2 How to Evaluate — Example for Task 1

```python
from datasets import load_dataset
from pipelines.shared.evaluate import (
    compute_auroc, compute_auprc,
    compute_f1_at_best_threshold, compute_ece
)
import pandas as pd

# Pull results and labels from Hub
t1 = load_dataset('factshield-team/results',
                   'task1_scores')['train'].to_pandas()
labels = load_dataset('factshield-team/faithbench-labels')['train'].to_pandas()

# Filter to one (summarizer, dataset, score_mode) condition
subset = t1[(t1.summarizer == 'bart') &
            (t1.dataset == 'cnndm') &
            (t1.score_mode == 'nli')]
merged = subset.merge(labels, on=['doc_id', 'sent_id'])

val  = merged[merged.split == 'val']
test = merged[merged.split == 'test']

auroc = compute_auroc(test['score'], test['label_binary'])
f1, threshold = compute_f1_at_best_threshold(
    val['score'],  val['label_binary'],
    test['score'], test['label_binary']
)

print(f"AUROC: {auroc:.4f}  |  F1: {f1:.4f}  |  Threshold: {threshold:.4f}")
```

---

## 10. Phase 5 — Final Comparison Table

**Owner: Manik assembles. Surya and Nischay supply their results CSVs.**  
**Weeks: 6-7**

### 10.1 Pull All Results and Assemble

Create `scripts/assemble_final_table.py`:

```python
from datasets import load_dataset, Dataset
from pipelines.shared.evaluate import compute_auroc, compute_auprc
import pandas as pd

# Pull all results from Hub
t1 = load_dataset('factshield-team/results', 'task1_scores')['train'].to_pandas()
t2 = load_dataset('factshield-team/results', 'task2_scores')['train'].to_pandas()
t3 = load_dataset('factshield-team/results', 'task3_scores')['train'].to_pandas()
labels = load_dataset('factshield-team/faithbench-labels')['train'].to_pandas()

# Standardise method names
t1['method'] = 'selfcheck_' + t1['score_mode']
t3['method'] = 'entropy_classifier'

# Combine
all_results = pd.concat([t1, t2, t3], ignore_index=True)
all_results = all_results.merge(labels, on=['doc_id', 'sent_id'])

# Compute metrics per (method, summarizer, dataset)
rows = []
for (method, model, ds), grp in all_results.groupby(
        ['method', 'summarizer', 'dataset']):
    rows.append({
        'method':     method,
        'summarizer': model,
        'dataset':    ds,
        'AUROC':      compute_auroc(grp['score'], grp['label_binary']),
        'AUPRC':      compute_auprc(grp['score'], grp['label_binary']),
        'n_samples':  len(grp),
    })

final_table = pd.DataFrame(rows).sort_values(
    ['dataset', 'summarizer', 'AUROC'], ascending=[True, True, False]
)
print(final_table.to_string(index=False))

# Push to Hub
Dataset.from_pandas(final_table).push_to_hub(
    'factshield-team/results',
    config_name='final_table',
    private=True
)
```

---

## 11. Phase 6 — Integration and Delivery

**Owner: All members.**  
**Weeks: 7-8**

### 11.1 What Lives Where — Final Inventory

| Location | Contents |
|---|---|
| `github.com/<org>/factshield` | All Python scripts, configs, tests, docs |
| `factshield-team/cache` (HF Dataset) | 27 configs: splits, summaries, K-samples, token scores |
| `factshield-team/results` (HF Dataset) | task1_scores, task2_scores, task3_scores, final_table |
| `factshield-team/faithbench-labels` (HF Dataset) | Typed annotation labels |
| `factshield-team/models` (HF Model) | task3_classifier.pkl |
| Weights and Biases | All experiment run logs and metrics |

### 11.2 End-to-End Validation

Run this from a clean environment to confirm full reproducibility:

```python
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
import joblib

login(token='hf_YOUR_TOKEN')

# 1. Verify all dataset splits exist
for config in ['cnndm_splits', 'xsum_splits', 'faithbench_splits']:
    ds = load_dataset('factshield-team/cache', config)
    assert len(ds['train']) > 0, f"Missing or empty: {config}"

# 2. Verify all cache configs exist (9 models x datasets x 3 types = 27)
for model in ['bart', 't5', 'pegasus']:
    for ds_name in ['cnndm', 'xsum', 'faithbench']:
        for suffix in ['summaries', 'k_samples', 'token_scores']:
            c = load_dataset('factshield-team/cache',
                              f'{model}_{ds_name}_{suffix}')
            assert len(c['train']) > 0

# 3. Verify all task results exist
for config in ['task1_scores', 'task2_scores', 'task3_scores', 'final_table']:
    ds = load_dataset('factshield-team/results', config)
    assert len(ds['train']) > 0, f"Missing or empty: {config}"

# 4. Verify trained model loads
path = hf_hub_download(repo_id='factshield-team/models',
                        filename='task3_classifier.pkl')
clf = joblib.load(path)
assert hasattr(clf, 'predict'), "Classifier did not load correctly"

# 5. Run full test suite
# pytest tests/ -v

print("All validation checks passed.")
```

### 11.3 Final PR Checklist

Before merging your final PR, confirm:

- All pipeline scripts committed to GitHub
- Results pushed to `factshield-team/results` on Hub
- All unit tests passing locally and in CI
- All plots saved as PNG and pushed to Hub results repo
- Your section of `docs/final_report.md` written and reviewed
- Experimental decisions logged in `docs/experimental_notes.md`

### 11.4 Final Report Structure

Write `docs/final_report.md` with this structure:

```
1. Abstract (150 words)
2. Introduction — hallucination problem, why three paradigms
3. Related Work — all 8 cited papers
4. Methods
   4.1 Shared setup (models, datasets, splits, decoding)
   4.2 Pipeline 1: SelfCheckGPT
   4.3 Pipeline 2: Entailment Verification
   4.4 Pipeline 3: Token Probability
5. Experimental Setup — HuggingFace infrastructure, compute budget
6. Results — Final Cross-Method Comparison Table, all plots
7. Analysis — research question answers, failure modes, entity analysis
8. Conclusion — which paradigm wins, limitations, future work
9. References
```

---

## 12. Week-by-Week Timeline

| Week | Surya | Nischay | Manik |
|---|---|---|---|
| Pre-W1 | HF account, join org, clone Space | HF account, join org, clone Space | Create HF org, 4 repos, 4 Spaces |
| W1 | Scaffold task1 Space, mock tests | Push dataset splits and FaithBench labels to Hub | Generate BART and T5 cache, push to Hub |
| W2 | V1: three scoring modes, push task1_scores | V1: SummaC-ZS and Conv, push task2_scores | Generate PEGASUS cache, V1 perplexity |
| W3 | V2: K ablation, latency plots | V2: MiniCheck, V3: AlignScore and ensemble | V2: entropy, V3: logistic classifier, push model |
| W4 | V3: cross-domain, push final task1 results | V4: type disaggregation, entity recall | V4: entity entropy, push all task3 results |
| W5 | Evaluate, calibration curves, plots | Calibration, correlation matrix, plots | Evaluation, entropy-threshold curves |
| W6 | Final PR, verify task1 results on Hub | Final PR, verify task2 results on Hub | Assemble final_table, push to Hub |
| W7 | Review Task 2 PR, write analysis section | Review Task 3 PR, write analysis section | Review Task 1 PR, end-to-end validation |
| W8 | Final report contribution, delivery | Final report contribution, delivery | Export W&B logs, tag release, delivery |

---

*Total cost: $0. Everything runs on HuggingFace free tier, Kaggle, and GitHub.*
