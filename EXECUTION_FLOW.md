# FactShield — Full Execution Flow
**Platform:** Local machine + Kaggle (compute) + HuggingFace Hub (storage)  
**Cost:** $0  
**Repo tree:** already set up ✓

---

## Quick reference — what runs where

| Phase | Where | Who |
|---|---|---|
| Phase 1 — data ingestion | Local | Nischay |
| Phase 2 — cache generation | **Kaggle** (GPU) | Manik |
| Phase 3 — pipelines | Local (load from Hub) | Surya / Nischay / Manik |
| Phase 4 — evaluation | Local | Each member |
| Phase 5 — final table | Local | Manik |

---

## Pre-flight — one-time setup (all members)

### 1. Clone repo and install deps

```bash
git clone https://github.com/<org>/factshield.git
cd factshield
conda activate fs_env          # or your venv
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Create `.env` in project root

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### 3. Add `__init__.py` so imports work

```bash
touch pipelines/__init__.py
touch pipelines/shared/__init__.py
touch pipelines/selfcheck/__init__.py
touch pipelines/entailment/__init__.py
touch pipelines/token_prob/__init__.py
```

### 4. Add to `.gitignore`

```
.env
data/
models/
__pycache__/
*.pkl
*.pyc
.ipynb_checkpoints/
```

---

## Phase 1 — Data Ingestion (Local, Nischay leads, Week 1)

### Step 1.1 — `pipelines/shared/data_loader.py`

Run this once to download and save all raw data locally.

```python
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from huggingface_hub import login

load_dotenv()
login(token=os.environ['HF_TOKEN'])

os.makedirs('data/raw', exist_ok=True)

# CNN/DailyMail
cnndm = load_dataset('cnn_dailymail', '3.0.0')
cnndm.save_to_disk('data/raw/cnndm')
print(f"✓ CNN/DailyMail: {cnndm}")

# XSUM
xsum = load_dataset('EdinburghNLP/xsum')
xsum.save_to_disk('data/raw/xsum')
print(f"✓ XSUM: {xsum}")

# FaithBench (GitHub CSV — not on HF Hub)
faithbench_url = 'https://raw.githubusercontent.com/vectara/FaithBench/main/FaithBench.csv'
faithbench = pd.read_csv(faithbench_url)
faithbench.to_csv('data/raw/faithbench.csv', index=False)
print(f"✓ FaithBench: {faithbench.shape}")
```

```bash
python3 pipelines/shared/data_loader.py
```

**Output:** `data/raw/cnndm/`, `data/raw/xsum/`, `data/raw/faithbench.csv`

---

### Step 1.2 — `pipelines/shared/splitter.py`

Run after data_loader. Creates 70/15/15 splits and pushes to HF Hub.

```python
import os
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import DatasetDict, Dataset, load_from_disk
from huggingface_hub import login

load_dotenv()
login(token=os.environ['HF_TOKEN'])

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def make_splits(dataset, split_key='test', n_sample=2000, seed=SEED):
    base = dataset[split_key]
    n_sample = min(n_sample, len(base))
    sampled = base.shuffle(seed=seed).select(range(n_sample))
    first_split = sampled.train_test_split(test_size=0.30, seed=seed)
    val_test = first_split['test'].train_test_split(test_size=0.50, seed=seed)
    return DatasetDict({
        'train': first_split['train'],   # 70% (~1400)
        'val':   val_test['train'],      # 15% (~300)
        'test':  val_test['test'],       # 15% (~300)
    })


def make_splits_from_dataset(dataset, seed=SEED):
    first_split = dataset.train_test_split(test_size=0.30, seed=seed)
    val_test = first_split['test'].train_test_split(test_size=0.50, seed=seed)
    return DatasetDict({
        'train': first_split['train'],   # 70% (~560)
        'val':   val_test['train'],      # 15% (~120)
        'test':  val_test['test'],       # 15% (~120)
    })


# Load from disk
cnndm = load_from_disk('data/raw/cnndm')
xsum  = load_from_disk('data/raw/xsum')
faithbench_df = pd.read_csv('data/raw/faithbench.csv')

# Split
cnndm_splits = make_splits(cnndm, split_key='test', n_sample=2000)
xsum_splits  = make_splits(xsum,  split_key='test', n_sample=2000)

faithbench_ds = Dataset.from_pandas(faithbench_df, preserve_index=False)
fb_splits     = make_splits_from_dataset(faithbench_ds)

print(f"✓ CNN/DM splits:     {cnndm_splits}")
print(f"✓ XSUM splits:       {xsum_splits}")
print(f"✓ FaithBench splits: {fb_splits}")

# Save locally
os.makedirs('data/splits', exist_ok=True)
cnndm_splits.save_to_disk('data/splits/cnndm')
xsum_splits.save_to_disk('data/splits/xsum')
fb_splits.save_to_disk('data/splits/faithbench')
print("✓ Splits saved to data/splits/")

# Push to HF Hub
cnndm_splits.push_to_hub('factshield-team/cache', config_name='cnndm_splits',      private=True)
xsum_splits.push_to_hub('factshield-team/cache',  config_name='xsum_splits',       private=True)
fb_splits.push_to_hub('factshield-team/cache',    config_name='faithbench_splits',  private=True)
print("✓ All splits pushed to factshield-team/cache")
```

```bash
python3 pipelines/shared/splitter.py
```

---

### Step 1.3 — `pipelines/shared/build_faithbench_labels.py`

Builds binary hallucination labels from FaithBench and pushes to Hub.

```python
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import login

load_dotenv()
login(token=os.environ['HF_TOKEN'])

df = pd.read_csv('data/raw/faithbench.csv')
print(f"Columns: {list(df.columns)}")   # source, summary, LLM, worst-label, best-label

HALLUCINATED = {'Unwanted', 'Questionable'}

typed_labels = pd.DataFrame({
    'doc_id':             df.index.astype(str),
    'source':             df['source'],
    'summary':            df['summary'],
    'llm':                df['LLM'],
    'worst_label':        df['worst-label'],
    'best_label':         df['best-label'],
    'label_binary_worst': df['worst-label'].map(lambda x: 1 if x in HALLUCINATED else 0),
    'label_binary_best':  df['best-label'].map(lambda x: 1 if x == 'Unwanted' else 0),
})

# Validate
assert typed_labels['doc_id'].notna().all(),                 "null doc_ids"
assert typed_labels['label_binary_worst'].isin([0,1]).all(), "invalid worst labels"
assert typed_labels['label_binary_best'].isin([0,1]).all(),  "invalid best labels"
assert len(typed_labels) == 800,                             f"expected 800, got {len(typed_labels)}"
print("✓ Validation passed")
print(typed_labels['label_binary_worst'].value_counts())

Dataset.from_pandas(typed_labels, preserve_index=False).push_to_hub(
    'factshield-team/faithbench-labels', private=True
)
print("✓ Typed labels pushed to factshield-team/faithbench-labels")
```

```bash
python3 pipelines/shared/build_faithbench_labels.py
```

---

### Step 1.4 — `tests/test_data.py`

```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset


def test_split_sizes():
    splits = load_dataset('factshield-team/cache', 'cnndm_splits')
    total = len(splits['train']) + len(splits['val']) + len(splits['test'])
    assert abs(len(splits['train']) / total - 0.70) < 0.02


def test_faithbench_labels():
    labels = load_dataset('factshield-team/faithbench-labels')['train']
    df = labels.to_pandas()
    assert len(df) == 800
    assert df['label_binary_worst'].isin([0, 1]).all()
    assert df['label_binary_best'].isin([0, 1]).all()
    assert df['doc_id'].notna().all()


def test_seed_reproducibility():
    from pipelines.shared.splitter import make_splits
    cnndm = load_dataset('cnn_dailymail', '3.0.0')
    s1 = make_splits(cnndm, seed=42)
    s2 = make_splits(cnndm, seed=42)
    assert list(s1['train']['id']) == list(s2['train']['id'])
```

```bash
pytest tests/test_data.py -v
# All 3 must pass before Phase 2 begins
```

---

## Phase 2 — Summarization Cache (Kaggle, Manik leads, Weeks 1–2)

> **Why Kaggle?** PEGASUS-large and K=10 sampling are too slow on CPU.  
> Kaggle gives a free T4 GPU (30 hrs/week). All output is pushed straight to HF Hub.

### Step 2.1 — Kaggle notebook setup

Create a new Kaggle notebook. In **Settings → Accelerator** select **GPU T4 x2**.  
Add your HF token as a Kaggle Secret named `HF_TOKEN`.

Paste this at the top of every Kaggle notebook cell:

```python
# Cell 1 — install deps
!pip install -q transformers datasets huggingface_hub python-dotenv

import os
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
login(token=HF_TOKEN)
print("✓ Logged in")
```

```python
# Cell 2 — load splits from Hub
from datasets import load_dataset

cnndm_splits = load_dataset('factshield-team/cache', 'cnndm_splits')
xsum_splits  = load_dataset('factshield-team/cache', 'xsum_splits')
fb_splits    = load_dataset('factshield-team/cache', 'faithbench_splits')
print("✓ Splits loaded")
```

---

### Step 2.2 — `pipelines/shared/generate.py`

This is the file Manik runs on Kaggle. Copy its contents into a Kaggle notebook cell.

```python
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = {
    'bart':    'facebook/bart-base',
    't5':      'google/t5-small',
    'pegasus': 'google/pegasus-large',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


def load_model(name):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODELS[name], torch_dtype=torch.float16
    ).eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[name])
    return model, tokenizer


def generate_summaries(model, tokenizer, dataset, model_name, dataset_name):
    """Single summary per doc. Used by Pipelines 2 and 3."""
    results = []
    for i, sample in enumerate(dataset):
        article_key = 'article' if 'article' in sample else 'document'
        id_key      = 'id' if 'id' in sample else 'doc_id'
        inputs = tokenizer(
            sample[article_key], return_tensors='pt',
            truncation=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs, num_beams=4,
                max_new_tokens=128, no_repeat_ngram_size=3
            )
        summary = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({
            'doc_id':       str(sample[id_key]),
            'source_text':  sample[article_key],
            'summary':      summary,
        })
        if i % 100 == 0 and i > 0:
            Dataset.from_list(results).push_to_hub(
                'factshield-team/cache',
                config_name=f'{model_name}_{dataset_name}_summaries',
                private=True
            )
            print(f"  checkpoint @ {i}")
    Dataset.from_list(results).push_to_hub(
        'factshield-team/cache',
        config_name=f'{model_name}_{dataset_name}_summaries',
        private=True
    )
    print(f"✓ {model_name}_{dataset_name}_summaries pushed ({len(results)} docs)")


def generate_k_samples(model, tokenizer, dataset, model_name, dataset_name, K=10):
    """K stochastic samples per doc. Used by Pipeline 1 only."""
    results = []
    for i, sample in enumerate(dataset):
        article_key = 'article' if 'article' in sample else 'document'
        id_key      = 'id' if 'id' in sample else 'doc_id'
        inputs = tokenizer(
            sample[article_key], return_tensors='pt',
            truncation=True, max_length=512
        ).to(DEVICE)
        samples = []
        for _ in range(K):
            with torch.no_grad():
                out = model.generate(
                    **inputs, do_sample=True,
                    temperature=1.0, max_new_tokens=128
                )
            samples.append(tokenizer.decode(out[0], skip_special_tokens=True))
        row = {'doc_id': str(sample[id_key])}
        for k, s in enumerate(samples):
            row[f'summary_k{k+1}'] = s
        results.append(row)
        if i % 100 == 0 and i > 0:
            Dataset.from_list(results).push_to_hub(
                'factshield-team/cache',
                config_name=f'{model_name}_{dataset_name}_k_samples',
                private=True
            )
            print(f"  checkpoint @ {i}")
    Dataset.from_list(results).push_to_hub(
        'factshield-team/cache',
        config_name=f'{model_name}_{dataset_name}_k_samples',
        private=True
    )
    print(f"✓ {model_name}_{dataset_name}_k_samples pushed ({len(results)} docs)")


def generate_with_scores(model, tokenizer, dataset, model_name, dataset_name):
    """Token log-probs per doc. Used by Pipeline 3 only."""
    results = []
    for i, sample in enumerate(dataset):
        article_key = 'article' if 'article' in sample else 'document'
        id_key      = 'id' if 'id' in sample else 'doc_id'
        inputs = tokenizer(
            sample[article_key], return_tensors='pt',
            truncation=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=1,          # greedy — beam search doesn't return per-token scores cleanly
                max_new_tokens=128
            )
        scores    = torch.stack(out.scores, dim=1)       # (1, seq_len, vocab)
        log_probs = scores.log_softmax(-1)
        token_ids = out.sequences[0, 1:]
        chosen_lp = log_probs[0, range(len(token_ids)), token_ids]
        results.append({
            'doc_id':         str(sample[id_key]),
            'token_logprobs': chosen_lp.cpu().numpy().tobytes(),
            'seq_len':        len(token_ids),
        })
        if i % 100 == 0 and i > 0:
            Dataset.from_list(results).push_to_hub(
                'factshield-team/cache',
                config_name=f'{model_name}_{dataset_name}_token_scores',
                private=True
            )
            print(f"  checkpoint @ {i}")
    Dataset.from_list(results).push_to_hub(
        'factshield-team/cache',
        config_name=f'{model_name}_{dataset_name}_token_scores',
        private=True
    )
    print(f"✓ {model_name}_{dataset_name}_token_scores pushed ({len(results)} docs)")
```

---

### Step 2.3 — Kaggle run order (one notebook per model)

Run three separate Kaggle notebooks — one per model — to stay within the 60-min GPU session limit.

**Notebook 1 — BART** (paste generate.py contents first, then):

```python
model, tokenizer = load_model('bart')

for ds_name, ds_splits in [
    ('cnndm',      cnndm_splits),
    ('xsum',       xsum_splits),
    ('faithbench', fb_splits),
]:
    split = ds_splits['train']
    generate_summaries(model, tokenizer, split, 'bart', ds_name)
    generate_k_samples(model, tokenizer, split, 'bart', ds_name, K=10)
    generate_with_scores(model, tokenizer, split, 'bart', ds_name)

print("✓ BART done")
```

**Notebook 2 — T5** (same structure, swap `'bart'` → `'t5'`)

**Notebook 3 — PEGASUS** (same structure, swap `'bart'` → `'pegasus'`)

After all three notebooks finish you will have **27 configs** in `factshield-team/cache`:
```
bart_cnndm_summaries        t5_cnndm_summaries        pegasus_cnndm_summaries
bart_cnndm_k_samples        t5_cnndm_k_samples        pegasus_cnndm_k_samples
bart_cnndm_token_scores     t5_cnndm_token_scores     pegasus_cnndm_token_scores
bart_xsum_summaries         t5_xsum_summaries         pegasus_xsum_summaries
...                         ...                        ...
```

---

## Phase 3 — Pipelines (Local, Weeks 2–4)

> Everyone loads from HF Hub. Nobody runs summarization locally.

### Pipeline 1 — SelfCheckGPT (Surya)

#### `pipelines/selfcheck/pipeline.py`

```python
import os
import time
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from huggingface_hub import login

load_dotenv()
login(token=os.environ['HF_TOKEN'])


def run_pipeline(model_name='bart', dataset_name='cnndm', K=5, score_mode='all'):
    """
    score_mode: 'bert' | 'nli' | 'ngram' | 'all'
    """
    from pipelines.selfcheck.scorer import score_bert, score_nli, score_ngram

    # Load K-samples cache from Hub
    k_cache = load_dataset(
        'factshield-team/cache',
        f'{model_name}_{dataset_name}_k_samples'
    )['train']
    k_df = k_cache.to_pandas()

    results = []
    start   = time.time()

    for _, row in k_df.iterrows():
        samples   = [row[f'summary_k{k}'] for k in range(1, K + 1)]
        primary   = samples[0]
        others    = samples[1:]
        sentences = [s.strip() for s in primary.split('.') if s.strip()]

        modes = ['bert', 'nli', 'ngram'] if score_mode == 'all' else [score_mode]

        for i, sent in enumerate(sentences):
            for mode in modes:
                if mode == 'bert':
                    score = score_bert(sent, sentences, others)
                elif mode == 'nli':
                    score = score_nli(sent, sentences, others)
                else:
                    score = score_ngram(sent, sentences, others)

                results.append({
                    'doc_id':     row['doc_id'],
                    'sent_id':    i,
                    'summarizer': model_name,
                    'dataset':    dataset_name,
                    'K':          K,
                    'score_mode': mode,
                    'score':      score,
                })

    elapsed = time.time() - start
    print(f"✓ {len(results)} sentence scores in {elapsed:.1f}s")
    return results, elapsed
```

#### `pipelines/selfcheck/scorer.py`

```python
from selfcheckgpt.modeling_selfcheck import (
    SelfCheckBERTScore,
    SelfCheckNLI,
    SelfCheckNgram,
)

_scorer_bert  = None
_scorer_nli   = None
_scorer_ngram = None


def _get_bert():
    global _scorer_bert
    if _scorer_bert is None:
        _scorer_bert = SelfCheckBERTScore(rescale_with_baseline=True)
    return _scorer_bert


def _get_nli():
    global _scorer_nli
    if _scorer_nli is None:
        _scorer_nli = SelfCheckNLI(nli_model='cross-encoder/nli-deberta-v3-large')
    return _scorer_nli


def _get_ngram():
    global _scorer_ngram
    if _scorer_ngram is None:
        _scorer_ngram = SelfCheckNgram(n=1)
    return _scorer_ngram


def score_bert(sentence, sentences, others):
    scores = _get_bert().predict(sentences=sentences, sampled_passages=others)
    idx = sentences.index(sentence)
    return float(scores[idx])


def score_nli(sentence, sentences, others):
    scores = _get_nli().predict(sentences=sentences, sampled_passages=others)
    idx = sentences.index(sentence)
    return float(scores[idx])


def score_ngram(sentence, sentences, others):
    scores = _get_ngram().predict(sentences=sentences, sampled_passages=others)
    idx = sentences.index(sentence)
    return float(scores[idx])
```

#### `pipelines/selfcheck/run.py`

```python
import argparse
import yaml
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import login
from pipelines.selfcheck.pipeline import run_pipeline

load_dotenv()
login(token=os.environ['HF_TOKEN'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name   = cfg['model_name']      # e.g. 'bart'
    dataset_name = cfg['dataset']         # e.g. 'cnndm'
    K            = cfg.get('K', 5)
    score_mode   = cfg.get('score_mode', 'all')

    results, elapsed = run_pipeline(model_name, dataset_name, K, score_mode)

    df = pd.DataFrame(results)
    df['latency_ms'] = (elapsed * 1000) / len(df['doc_id'].unique())
    df['split'] = 'test'

    Dataset.from_pandas(df, preserve_index=False).push_to_hub(
        'factshield-team/results',
        config_name='task1_scores',
        private=True
    )
    print("✓ task1_scores pushed to factshield-team/results")


if __name__ == '__main__':
    main()
```

**Config file — `configs/selfcheck_scoring_sweep.yaml`:**

```yaml
model_name: bart
dataset: cnndm
K: 5
score_mode: all
seed: 42
wandb_run_name: task1_bart_cnndm_v1_k5
hf_cache_repo: factshield-team/cache
hf_results_repo: factshield-team/results
```

**Run V1:**

```bash
python3 -m pipelines.selfcheck.run --config configs/selfcheck_scoring_sweep.yaml
```

**Run V2 (K ablation) — `configs/selfcheck_k_ablation.yaml`:**

```yaml
model_name: bart
dataset: cnndm
K: 10          # change to 3, 5, 7, 10 in separate runs
score_mode: nli
seed: 42
```

**Run V3 (cross-domain) — `configs/selfcheck_crossdomain.yaml`:**

```yaml
model_name: bart
dataset: xsum   # change to faithbench for second cross-domain run
K: 5
score_mode: nli
seed: 42
```

---

### Pipeline 2 — Entailment Verification (Nischay)

#### `pipelines/entailment/scorer.py`

```python
from summac.model_summac import SummaCZS, SummaCConv

_zs   = None
_conv = None


def _get_zs():
    global _zs
    if _zs is None:
        _zs = SummaCZS(granularity='sentence', model_name='vitc')
    return _zs


def _get_conv():
    global _conv
    if _conv is None:
        _conv = SummaCConv(models=['vitc'], bins='percentile', granularity='sentence')
    return _conv


def score_summac_zs(source, sentence):
    return float(_get_zs().score([source], [sentence])['scores'][0])


def score_summac_conv(source, sentence):
    return float(_get_conv().score([source], [sentence])['scores'][0])


def score_minicheck(source, sentence):
    from minicheck.minicheck import MiniCheck
    scorer = MiniCheck(model_name='roberta-large')
    _, score, _, _ = scorer.score(docs=[source], claims=[sentence])
    return float(score[0])


def score_alignscore(source, sentence):
    from alignscore import AlignScore
    scorer = AlignScore(
        model='roberta-large', batch_size=32, device='cpu',
        ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt'
    )
    return float(scorer.score(contexts=[source], claims=[sentence])[0])
```

#### `pipelines/entailment/pipeline.py`

```python
import time
import pandas as pd
from datasets import load_dataset
from pipelines.entailment.scorer import (
    score_summac_zs, score_summac_conv,
    score_minicheck, score_alignscore
)


def run_pipeline(model_name='bart', dataset_name='cnndm', methods=None):
    if methods is None:
        methods = ['summac_zs', 'summac_conv']

    cache = load_dataset(
        'factshield-team/cache',
        f'{model_name}_{dataset_name}_summaries'
    )['train']
    df = cache.to_pandas()

    results = []
    start   = time.time()

    for _, row in df.iterrows():
        sentences = [s.strip() for s in row['summary'].split('.') if s.strip()]
        source    = row['source_text']

        for i, sent in enumerate(sentences):
            for method in methods:
                if method == 'summac_zs':
                    score = score_summac_zs(source, sent)
                elif method == 'summac_conv':
                    score = score_summac_conv(source, sent)
                elif method == 'minicheck':
                    score = score_minicheck(source, sent)
                elif method == 'alignscore':
                    score = score_alignscore(source, sent)
                else:
                    raise ValueError(f"Unknown method: {method}")

                results.append({
                    'doc_id':     row['doc_id'],
                    'sent_id':    i,
                    'summarizer': model_name,
                    'dataset':    dataset_name,
                    'method':     method,
                    'score':      score,
                })

    elapsed = time.time() - start
    print(f"✓ {len(results)} scores in {elapsed:.1f}s")
    return results, elapsed
```

#### `pipelines/entailment/run.py`

```python
import argparse
import yaml
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import login
from pipelines.entailment.pipeline import run_pipeline

load_dotenv()
login(token=os.environ['HF_TOKEN'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results, elapsed = run_pipeline(
        model_name=cfg['model_name'],
        dataset_name=cfg['dataset'],
        methods=cfg.get('methods', ['summac_zs', 'summac_conv'])
    )

    df = pd.DataFrame(results)
    df['latency_ms'] = (elapsed * 1000) / len(df['doc_id'].unique())
    df['split'] = 'test'

    Dataset.from_pandas(df, preserve_index=False).push_to_hub(
        'factshield-team/results',
        config_name='task2_scores',
        private=True
    )
    print("✓ task2_scores pushed to factshield-team/results")


if __name__ == '__main__':
    main()
```

**Run V1 (SummaC):**

```bash
python3 -m pipelines.entailment.run --config configs/entailment_summac.yaml
```

`configs/entailment_summac.yaml`:

```yaml
model_name: bart
dataset: cnndm
methods: [summac_zs, summac_conv]
seed: 42
```

**Run V2 (MiniCheck):**

```bash
python3 -m pipelines.entailment.run --config configs/entailment_minicheck.yaml
```

**Run V3 (AlignScore):**

```bash
python3 -m pipelines.entailment.run --config configs/entailment_alignscore.yaml
```

---

### Pipeline 3 — Token Probability (Manik)

#### `pipelines/token_prob/scorer.py`

```python
import math
import numpy as np


def compute_perplexity(log_probs: np.ndarray) -> float:
    """Sentence-level perplexity from token log-probs."""
    return math.exp(-np.mean(log_probs))


def compute_entropy(log_probs: np.ndarray) -> tuple[float, float]:
    """
    Mean and max token entropy from log-probs.
    log_probs shape: (seq_len,) — chosen token log-probs only.
    This is an approximation; full vocab distribution gives exact entropy.
    """
    probs = np.exp(log_probs)
    # Approximate: treat chosen prob as dominant; entropy proxy
    entropy_per_token = -(probs * log_probs)
    return float(np.mean(entropy_per_token)), float(np.max(entropy_per_token))


def extract_features(row) -> dict:
    log_probs  = np.frombuffer(row['token_logprobs'], dtype=np.float32)
    perplexity = compute_perplexity(log_probs)
    mean_ent, max_ent = compute_entropy(log_probs)
    tail_mean_nll = float(-np.mean(log_probs[-5:]))  # last 5 tokens
    return {
        'perplexity':    perplexity,
        'mean_entropy':  mean_ent,
        'max_entropy':   max_ent,
        'tail_mean_nll': tail_mean_nll,
        'sent_length':   int(row['seq_len']),
    }
```

#### `pipelines/token_prob/classifier.py`

```python
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from huggingface_hub import HfApi

FEATURE_COLS = ['perplexity', 'mean_entropy', 'max_entropy',
                'tail_mean_nll', 'sent_length']


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> CalibratedClassifierCV:
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['label_binary_worst'].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df['label_binary_worst'].values

    best_C, best_auroc = 1.0, 0.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, penalty='l2', max_iter=1000)
        cv  = cross_val_score(clf, X_train, y_train,
                              cv=StratifiedKFold(5), scoring='roc_auc')
        print(f"  C={C:.2f} → AUROC {cv.mean():.4f} ± {cv.std():.4f}")
        if cv.mean() > best_auroc:
            best_auroc, best_C = cv.mean(), C

    print(f"✓ Best C={best_C}, val AUROC={best_auroc:.4f}")
    final = LogisticRegression(C=best_C, penalty='l2', max_iter=1000)
    final.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(final, method='sigmoid', cv='prefit')
    calibrated.fit(X_val, y_val)
    return calibrated


def push_model(clf, token: str):
    joblib.dump(clf, 'task3_classifier.pkl')
    HfApi().upload_file(
        path_or_fileobj='task3_classifier.pkl',
        path_in_repo='task3_classifier.pkl',
        repo_id='factshield-team/models',
        repo_type='model',
        token=token
    )
    print("✓ Classifier pushed to factshield-team/models")


def load_model_from_hub():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id='factshield-team/models',
                           filename='task3_classifier.pkl')
    return joblib.load(path)
```

#### `pipelines/token_prob/pipeline.py`

```python
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from pipelines.token_prob.scorer import extract_features


def run_pipeline(model_name='bart', dataset_name='cnndm'):
    scores_ds = load_dataset(
        'factshield-team/cache',
        f'{model_name}_{dataset_name}_token_scores'
    )['train']
    scores_df = scores_ds.to_pandas()

    labels_ds = load_dataset('factshield-team/faithbench-labels')['train']
    labels_df = labels_ds.to_pandas()

    results = []
    start   = time.time()

    for _, row in scores_df.iterrows():
        feats = extract_features(row)
        feats.update({
            'doc_id':     row['doc_id'],
            'summarizer': model_name,
            'dataset':    dataset_name,
        })
        results.append(feats)

    elapsed = time.time() - start
    results_df = pd.DataFrame(results)

    # Merge with labels for classifier training
    merged = results_df.merge(labels_df[['doc_id', 'label_binary_worst',
                                          'label_binary_best']],
                               on='doc_id', how='left')
    print(f"✓ {len(results)} docs processed in {elapsed:.1f}s")
    return merged, elapsed
```

#### `pipelines/token_prob/run.py`

```python
import argparse
import yaml
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from huggingface_hub import login
from pipelines.token_prob.pipeline import run_pipeline
from pipelines.token_prob.classifier import train, push_model

load_dotenv()
login(token=os.environ['HF_TOKEN'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    merged, elapsed = run_pipeline(cfg['model_name'], cfg['dataset'])

    train_df = merged[merged['split'] == 'train'] if 'split' in merged else merged
    val_df   = merged[merged['split'] == 'val']   if 'split' in merged else merged.sample(frac=0.15, random_state=42)

    clf = train(train_df, val_df)

    # Score test set
    test_df = merged
    FEATURE_COLS = ['perplexity', 'mean_entropy', 'max_entropy',
                    'tail_mean_nll', 'sent_length']
    X_test = test_df[FEATURE_COLS].values
    test_df = test_df.copy()
    test_df['score']    = clf.predict_proba(X_test)[:, 1]
    test_df['split']    = 'test'
    test_df['method']   = 'token_prob_classifier'
    test_df['latency_ms'] = (elapsed * 1000) / len(test_df)

    push_model(clf, token=os.environ['HF_TOKEN'])

    Dataset.from_pandas(test_df, preserve_index=False).push_to_hub(
        'factshield-team/results',
        config_name='task3_scores',
        private=True
    )
    print("✓ task3_scores pushed to factshield-team/results")


if __name__ == '__main__':
    main()
```

**Run:**

```bash
python3 -m pipelines.token_prob.run --config configs/token_prob_classifier.yaml
```

---

## Phase 4 — Evaluation (Local, Weeks 5–6)

### `pipelines/shared/evaluate.py`

```python
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from netcal.metrics import ECE


def compute_auroc(scores, labels):
    return roc_auc_score(labels, scores)


def compute_auprc(scores, labels):
    return average_precision_score(labels, scores)


def compute_f1_at_best_threshold(val_scores, val_labels,
                                  test_scores, test_labels):
    """Optimise threshold on val, never on test."""
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
    return (end_time - start_time) * 1000 / n_queries   # ms per query
```

### Example evaluation call (same pattern for all three tasks)

```python
from datasets import load_dataset
from pipelines.shared.evaluate import (
    compute_auroc, compute_auprc,
    compute_f1_at_best_threshold, compute_ece
)

t1     = load_dataset('factshield-team/results', 'task1_scores')['train'].to_pandas()
labels = load_dataset('factshield-team/faithbench-labels')['train'].to_pandas()

subset = t1[(t1.summarizer == 'bart') &
            (t1.dataset    == 'cnndm') &
            (t1.score_mode == 'nli')]
merged = subset.merge(labels, on='doc_id')

val  = merged[merged.split == 'val']
test = merged[merged.split == 'test']

auroc        = compute_auroc(test['score'], test['label_binary_worst'])
f1, thresh   = compute_f1_at_best_threshold(
    val['score'],  val['label_binary_worst'],
    test['score'], test['label_binary_worst']
)
ece          = compute_ece(test['score'].values, test['label_binary_worst'].values)

print(f"AUROC: {auroc:.4f} | F1: {f1:.4f} | ECE: {ece:.4f} | threshold: {thresh:.4f}")
```

---

## Phase 5 — Final Table (Local, Manik, Weeks 6–7)

### `scripts/assemble_final_table.py`

```python
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from huggingface_hub import login
from pipelines.shared.evaluate import compute_auroc, compute_auprc

load_dotenv()
login(token=os.environ['HF_TOKEN'])

# Pull all results
t1     = load_dataset('factshield-team/results', 'task1_scores')['train'].to_pandas()
t2     = load_dataset('factshield-team/results', 'task2_scores')['train'].to_pandas()
t3     = load_dataset('factshield-team/results', 'task3_scores')['train'].to_pandas()
labels = load_dataset('factshield-team/faithbench-labels')['train'].to_pandas()

# Normalise method column
t1['method'] = 'selfcheck_' + t1['score_mode']
t2['method'] = t2['method']
t3['method'] = 'token_prob_classifier'

all_results = pd.concat([t1, t2, t3], ignore_index=True)
all_results = all_results.merge(
    labels[['doc_id', 'label_binary_worst']], on='doc_id', how='inner'
)

# Compute per (method, summarizer, dataset)
rows = []
for (method, model, ds), grp in all_results.groupby(['method', 'summarizer', 'dataset']):
    rows.append({
        'method':     method,
        'summarizer': model,
        'dataset':    ds,
        'AUROC':      compute_auroc(grp['score'], grp['label_binary_worst']),
        'AUPRC':      compute_auprc(grp['score'], grp['label_binary_worst']),
        'n_samples':  len(grp),
    })

final_table = pd.DataFrame(rows).sort_values(
    ['dataset', 'summarizer', 'AUROC'], ascending=[True, True, False]
)

print(final_table.to_string(index=False))

Dataset.from_pandas(final_table, preserve_index=False).push_to_hub(
    'factshield-team/results',
    config_name='final_table',
    private=True
)
print("✓ final_table pushed to factshield-team/results")
```

```bash
python3 scripts/assemble_final_table.py
```

---

## Phase 6 — CI and Delivery

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: pytest tests/ -v
```

> Add `HF_TOKEN` to your GitHub repo under **Settings → Secrets → Actions**.

### End-to-end validation

```python
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
import joblib, os
from dotenv import load_dotenv

load_dotenv()
login(token=os.environ['HF_TOKEN'])

# 1. Dataset splits
for config in ['cnndm_splits', 'xsum_splits', 'faithbench_splits']:
    ds = load_dataset('factshield-team/cache', config)
    assert len(ds['train']) > 0, f"Empty: {config}"
    print(f"✓ {config}")

# 2. All 27 cache configs
for model in ['bart', 't5', 'pegasus']:
    for ds_name in ['cnndm', 'xsum', 'faithbench']:
        for suffix in ['summaries', 'k_samples', 'token_scores']:
            c = load_dataset('factshield-team/cache', f'{model}_{ds_name}_{suffix}')
            assert len(c['train']) > 0
            print(f"✓ {model}_{ds_name}_{suffix}")

# 3. Task results
for config in ['task1_scores', 'task2_scores', 'task3_scores', 'final_table']:
    ds = load_dataset('factshield-team/results', config)
    assert len(ds['train']) > 0, f"Empty: {config}"
    print(f"✓ {config}")

# 4. Trained classifier
path = hf_hub_download(repo_id='factshield-team/models',
                        filename='task3_classifier.pkl')
clf = joblib.load(path)
assert hasattr(clf, 'predict')
print("✓ classifier loads correctly")

print("\n✅ All validation checks passed.")
```

---

## Week-by-week run order

| Week | Command / action |
|---|---|
| W1 | `python3 pipelines/shared/data_loader.py` |
| W1 | `python3 pipelines/shared/splitter.py` |
| W1 | `python3 pipelines/shared/build_faithbench_labels.py` |
| W1 | `pytest tests/test_data.py -v` — must be green |
| W1–2 | Manik: run 3 Kaggle notebooks (BART, T5, PEGASUS) |
| W2 | `python3 -m pipelines.selfcheck.run --config configs/selfcheck_scoring_sweep.yaml` |
| W2 | `python3 -m pipelines.entailment.run --config configs/entailment_summac.yaml` |
| W2 | `python3 -m pipelines.token_prob.run --config configs/token_prob_perplexity.yaml` |
| W3 | Repeat runs for remaining configs (K ablation, MiniCheck, entropy) |
| W4 | Cross-domain runs, AlignScore, classifier |
| W5–6 | Evaluate each pipeline, generate plots |
| W6 | `python3 scripts/assemble_final_table.py` |
| W7 | End-to-end validation script |
| W8 | Final report, tag release |
