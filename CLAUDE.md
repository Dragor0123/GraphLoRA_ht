# 📑 Program Design Document

**Dual Encoders (Separate Pretraining) for**\
**GraphFourierFT + (Heterophilic) GraphControl vs GraphLoRA +
(Heterophilic) GraphControl**

> This spec is written to be friendly to engineers new to GraphControl,
> GraphLoRA, and FourierFT.\
> It plugs into your current repo layout:
>
>     .
>     ├── config.yaml, config2.yaml
>     ├── main.py
>     ├── pre_train.py
>     ├── uitl.py
>     ├── (new files here if they are necessary)
>     └── model/
>         ├── GNN_model.py
>         ├── GRACE_model.py
>         ├── GraphLoRA.py, 
>         ├── GraphFourierFT.py
>         ├── GNN_FourierFT.py
>         ├── FourierFT_adapter.py 
>         └── (new files here if they are necessary)
>
> FourierFT adapters are used **only at fine-tuning**; never during
> pretraining.
------------------------------------------------------------------------
## 1) Objective

-   Implement **Strategy-1 (Dual Encoders, Separate Pretraining)** on
    top of the current codebase so that:
    -   We can **pretrain two encoders separately**:
        -   **Structural encoder (Flow-1):** positional-embedding--only,
            feature-free (GraphControl style).
        -   **Feature encoder (Flow-2):** GRACE style, feature-aware
            (already supported).
    -   We can **fine-tune on the same target graph** with either:
        -   **GraphFourierFT + Heterophilic GraphControl**, or
        -   **GraphLoRA + Heterophilic GraphControl**,
    -   ...and **compare** accuracy and memory efficiency fairly (same
        pretraining start point and same four losses).
------------------------------------------------------------------------
## 2) Key Requirements

1. **Loss consistency (4-loss):**

   $$
   \mathcal{L} = \lambda_1 L_{\text{cls}} + \lambda_2 L_{\text{smmd}} + \lambda_3 L_{\text{contrastive}} + \lambda_4 L_{\text{struct\_reg}}
   $$

   Use exactly the same formulations used in the current repo (already implemented in your transfer loops).

2. **Two independent pretraining flows:**
   - **Flow-1 (Structural):** feature-free, subgraph contrastive with Laplacian PE.
   - **Flow-2 (GRACE):** feature-aware, already implemented in `pre_train.py`.

3. **Adapter options at fine-tuning:**
   - Select **FourierFT** (weight-space spectral adapter) or **LoRA** (low-rank adapter) on the *same* feature encoder.

4. **Condition Generation + ControlNet:**
   - Role / Inverse / PPR condition → Laplacian PE → **ZeroMLP** → **FiLM** modulation over adapter parameters.

5. **Memory logging:**
   - Print **adapter-only** trainables and **total** trainables (adapter + projector + classifier, etc.) in counts and MB.

6. **Reproducibility:**
   - Keep seeds, split masks, and checkpoint naming consistent across both adapter variants.
------------------------------------------------------------------------
## 3) What is What (short glossary)

- **Structural encoder $g_{\theta_s}^*$:**  
  GAT-based encoder learned **without node features**; uses only positional embeddings of graphs/subgraphs (GraphControl style).  
  Frozen at fine-tuning.

- **Feature encoder $g_{\theta_f}^*$:**  
  GAT-based encoder learned with **GRACE** (feature & edge augmentations).  
  Frozen at fine-tuning; adapters are attached here.

- **FourierFT adapter:**  
  Parameter-efficient adapter that represents $\Delta W$ in the **weight spectrum** (train only spectral coefficients, reconstruct via IDFT).

- **LoRA branch:**  
  Parameter-efficient adapter that represents $\Delta W = BA$ (low-rank).

- **ControlNet (ZeroMLP):**  
  Tiny MLP that maps condition-PE to $(\gamma, \beta)$ to **modulate adapters** (FiLM).

- **Projector $\Pi$ (optional):**  
  Small linear module to align input or embedding dimensions across datasets/domains.
------------------------------------------------------------------------
## 4) End-to-End Workflow (Strategy-1)

### 4.1 Pretraining (two separate flows)

#### 4.1.1 Flow-1: Structural Pretraining (feature-free)

**Input:** source graphs $\{A_s\}$

**Process:**
- Compute **Laplacian eigenvectors** (PE) per graph; sign-align + L2-normalize.
- For each node, **subgraph sampling** (Random walk with restart) to make two views.
- Forward through **GAT backbone + small projection head**.
- Contrastive loss (InfoNCE) on subgraph instances.
- **Adapters OFF** (no LoRA/FourierFT here).

**Output:**  
`pre_trained_gnn/STRUCT.GAT.encoder_S.pth` (frozen later)

---

#### 4.1.2 Flow-2: GRACE Pretraining (feature-aware)

**Input:** source graphs $(A_f, X_f)$

**Process:**
- Standard **GRACE augmentations** (edge dropout, feature masking).
- Forward through **GAT backbone + GRACE projection head**.
- GRACE contrastive loss.
- **Adapters OFF**.

**Output:** (already produced by your pipeline)  
`pre_trained_gnn/{Dataset}.GRACE.GAT.True.pth` → rename/link to  
`pre_trained_gnn/GRACE.GAT.encoder_F.pth` for clarity.

>**Tip:** Ensure both encoders output the same **embedding dim** (e.g., 256) to simplify later fusion/losses.
------------------------------------------------------------------------
## 4.2 Fine-tuning (target graph)
### Step 0 — Choose adapter
- `--method {fourier|lora}` (already supported in `main.py`)
- Adapters attach to **feature encoder** $g_{\theta_f}^*$ only.

### Step 1 — Load and freeze encoders
- Load `encoder_S.pth` and `encoder_F.pth`.
- Freeze all their parameters.

### Step 2 — Build condition (Heterophilic GraphControl)
- Choose `--condition_type {role|inverse|ppr}`.
- Build **condition adjacency** $A_{\text{cond}}$ on the target graph (three options):
  - **role:** k-hop degree role similarity → top-quantile threshold.
  - **inverse:** feature cosine **bottom** quantile.
  - **ppr:** $PPR(A_t) \times$ feature distance → top quantile.
- From $A_{\text{cond}}$, compute Laplacian PE → **condition-PE** $P_{\text{cond}}$.


### Step 3 — ZeroMLP (ControlNet) → FiLM over adapters
- Input $P_{\text{cond}}$ (or $g_{\theta_s}^*(P_{\text{cond}})$) to ZeroMLP $\phi$.
- Obtain $\gamma, \beta = \phi(\cdot)$.
- **FourierFT:** modulate spectral coefficients $c' = \gamma \odot c + \beta$.  
- **LoRA:** modulate low-rank branch amplitude (e.g., $\Delta W' = \gamma \odot (BA) + \beta$).

### Step 4 — Forward & Heads
- Optional **Projector $\Pi$** for input or embedding alignment.
- Forward through frozen $g_{\theta_f}^*$ with the adapter active $\rightarrow z_{\text{adapt}}$.
- Classifier $h(z_{\text{adapt}})$ for logits.

### Step 5 — Losses (same for both adapters)
- $L_{\text{cls}}$: supervised cross-entropy (train mask).
- $L_{\text{smmd}}$: PPR-weighted alignment (target vs source).
- $L_{\text{contrastive}}$: consistency between base and adapted embeddings.
- $L_{\text{struct\_reg}}$: graph-aware regularizer on logits/adjacency.

### Step 6 — Optimize
- Three groups (separate LR/WD):
  1. Adapter params (**FourierFT** spectral coefficients or **LoRA** A/B).
  2. **ZeroMLP** (ControlNet).
  3. **Projector + Classifier**.
- Early-stop on validation accuracy.

### Step 7 — Logging
- Print **Adapter-only params** (count + MB).
- Print **Total trainables** (adapter + projector + classifier).
- Track accuracy (train/val/test) and the 4-loss curve.

------------------------------------------------------------------------
## 5) Modules to Modify / Add (with signatures)
### 5.1 `pre_train.py` — add a structural pretraining mode

- Add CLI flag: `--pretext {GRACE,STRUCT}` (default: GRACE)
- **New function:**
```python
def pretrain_structural(args, config) -> str:
    """
    Returns path to the saved structural-encoder checkpoint.
    Input  : datasets (A only), PE settings (k-eigs), subgraph sampler cfg
    Process: Laplacian PE -> subgraph contrastive (two views)
    Output : path 'pre_trained_gnn/STRUCT.GAT.encoder_S.pth'
    """
```
> Flow-2 (GRACE) path remains as your current pre_train.py training.

### 5.2 main.py — route both pretraining flows and both adapters
- New/confirm flags:
    * --pretext {GRACE,STRUCT}
    * --condition_type {role,inverse,ppr}
    * --method {lora,fourier} (already present)
    * --encoder_S_ckpt, --encoder_F_ckpt (optional explicit paths)
- High-level pseudo:

```python
if args.is_pretrain:
    if args.pretext == "STRUCT":
        path_S = pretrain_structural(args, config)
    elif args.pretext == "GRACE":
        path_F = pretrain_grace(args, config)  # existing flow
else:
    if args.method == "fourier":
        transfer_fourier(args, config, ...)
    else:
        transfer_lora(args, config, ...)
```

### 5.3 model/ConditionGen.py — (new) condition + ZeroMLP
```python
class ZeroMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=2*coeff_dim):
        """Outputs [gamma, beta] for FiLM."""
    def forward(self, P_cond): ...
```

```python
def build_condition_pe(data, condition_type, **cfg):
    """
    role    : k-hop role similarity -> threshold -> LapPE
    inverse : bottom-quantile feature cosine -> LapPE
    ppr     : PPR(A) * feat_distance -> top-quantile -> LapPE
    Returns P_cond (N x d_pe)
    """
```
>Call this from both GraphFourierFT.py and GraphLoRA.py to keep parity.

### 5.4 `model/GraphFourierFT.py` — adapter modulation + logging

- **Where:** inside `transfer_fourier(...)`
- **Add:**
  - Build $P_{\text{cond}} = \text{build\_condition\_pe}(...)$.
  - $(\gamma, \beta) = \text{ZeroMLP}(P_{\text{cond}})$.
  - Modulate Fourier coefficients $c' = \gamma \odot c + \beta$.
  - Ensure **4 losses** are computed exactly as now.
- **Memory logging:**

```python
print_memory_usage(gnn_fourier, adapter_module_names=['fourier', 'adapter'])
print_memory_usage(nn.ModuleList([projector, logreg, gnn_fourier]),
                   adapter_module_names=['fourier','adapter'])
```

### 5.5 `model/GraphLoRA.py` — same modulation + same logging

* Mirror the steps above, but modulation controls LoRA’s A/B branch magnitude (e.g., pre-activation scaling or residual weighting).
* Use the same `build_condition_pe` and `ZeroMLP` so **both adapters** share identical conditioning.

### 5.6 `util.py` — utilities you’ll likely extend
* `print_memory_usage(model_or_list, adapter_module_names=[])`
* Count params that match adapter names (e.g., `'fourier'`, `'adapter'`, or explicit class checks) and total trainables. Convert to MB via dtype size.
* Condition helpers (if not placed in `ConditionGen.py`): cosine similarity, quantile thresholding, Laplacian, PPR, sign-aligned eigenvectors, etc.
* Few-shot masks and PPR weights (already present): keep as is.

------------------------------------------------------------------------
## 6) Full “Input / Process / Output” Breakdown
### 6.1 Flow-1 Structural Pretraining
**Input:** graphs $\{A_s\}$

**Process:**
- Compute Laplacian $L = D - A$; eigen-decompose $L \rightarrow$ top-$k$ eigenvectors $P$.  
  Normalize: sign alignment + L2.
- For each node, sample two subgraphs (choose RWR).
- Forward: $P^{(1)}, P^{(2)} \rightarrow g_{\theta_s} \rightarrow \text{proj} \rightarrow \text{InfoNCE}$.
- Optimize $g_{\theta_s}$ and projection head; adapters **OFF**.
- Save **encoder only**.

**Output:** `encoder_S.pth` (frozen later)

---

### 6.2 Flow-2 GRACE Pretraining

**Input:** graphs $(A_f, X_f)$

**Process:** (already in your repo)
- Augment $X_f, A_f \rightarrow$ two views.
- Forward with $g_{\theta_f}$ + GRACE projection $\rightarrow \text{InfoNCE}$.
- Save **encoder only**.

**Output:** `encoder_F.pth` (frozen later)

### 6.3 Fine-Tuning with GraphFourierFT

**Input:** target $G_t = (A_t, X_t, Y_t)$, `encoder_S.pth`, `encoder_F.pth`

**Process:**
- Load → **freeze** both encoders.
- Attach **FourierFT adapters** to $g_{\theta_t}^*$ (selected layers).
- Build **condition adjacency** $A_{\text{cond}}$ by `condition_type`.
- Compute condition-PE $P_{\text{cond}}$.
- $(\gamma, \beta) = \text{ZeroMLP}(P_{\text{cond}}) \rightarrow$ modulate FourierFT coeff  
  $c' = \gamma \odot c + \beta$.
- (Optional) Projector $\Pi$ for dimension/domain alignment.
- Forward $X_t$ through $g_{\theta_t}^*$ + FourierFT $\rightarrow z_{\text{adapt}}$.
- Classifier $h(z_{\text{adapt}})$.
- Compute **4-loss** (cls, smmd, contrastive, struct\_reg).
- Optimize **{adapter, ZeroMLP, (Projector, Classifier)}** only.
- Log **adapter-only** and **total** trainables.

**Output:** best checkpoint (adapter + ZeroMLP + optional heads)

---

### 6.4 Fine-Tuning with GraphLoRA

**Input:** same as above

**Process:** identical to FourierFT path, except:
- Attach **LoRA branch** (rank $r$) to $g_{\theta_t}^*$.
- ZeroMLP outputs $(\gamma, \beta)$ modulate LoRA path (scale residual or A/B factors).
- Compute same **4-loss**, optimize same groups, log same memory lines.

**Output:** best checkpoint (LoRA branch + ZeroMLP + optional heads)

------------------------------------------------------------------------


## 7) Config & CLI (suggested)
- Add to `config.yaml`:
``` yaml
pretrain:
  struct:
    k_eigs: 32
    subgraph_type: "rwr"         # random walk with restart
    walk_length: 256             # number of steps per walk
    restart_prob: 0.8            # probability of restarting at the origin node
    num_subgraphs: 1024          # how many subgraphs to sample per epoch
    batch_size: 256              # how many subgraphs per batch
    epochs: 200
    lr: 1e-3
    out_dim: 256

  grace:
    # your existing GRACE block (kept as-is)
    out_dim: 256
```
- Example commands: 
```bash
# Flow-1 (Structural) pretraining
python main.py --is_pretrain True --pretext STRUCT --dataset Cora

# Flow-2 (GRACE) pretraining
python main.py --is_pretrain True --pretext GRACE --dataset Cora

# Fine-tune with FourierFT + GraphControl (role condition)
python main.py --is_transfer True --method fourier \
  --condition_type role --dataset Cora \
  --encoder_S_ckpt pre_trained_gnn/STRUCT.GAT.encoder_S.pth \
  --encoder_F_ckpt pre_trained_gnn/GRACE.GAT.encoder_F.pth

# Fine-tune with LoRA + GraphControl (same settings)
python main.py --is_transfer True --method lora \
  --condition_type role --dataset Cora \
  --encoder_S_ckpt pre_trained_gnn/STRUCT.GAT.encoder_S.pth \
  --encoder_F_ckpt pre_trained_gnn/GRACE.GAT.encoder_F.pth
```

------------------------------------------------------------------------
## 8) Evaluation Plan
- Datasets: Cora/CiteSeer/PubMed (homophilic), Chameleon/Squirrel/Actor (heterophilic).
- Few-shot settings: match your existing mask generator (e.g., 5-shot/10-shot).
- Report: accuracy ± std over 5–10 seeds, plus:
- Adapter-only param count/MB,
- Total trainables/MB,
- Validation curves,
- Ablations: `condition_type ∈ {role, inverse, ppr}, LoRA rank r, FourierFT coeff count n.`  
------------------------------------------------------------------------
## 9) Risks & Mitigations
- Iterator misuse in MMD/SMMD: don’t call next(iter(loader)) in a loop; reuse a single iterator or use full source reps.
- Eigenvector sign ambiguity: sign-align PE across graphs/subgraphs.
- Dimension mismatch: keep encoder output dims equal; use Projector only if necessary.
- Cost of PE on large graphs: cache PE to disk; warn for $N>10k$.
------------------------------------------------------------------------
## 10) Task Checklist (copy to issue/PR)
 - [ ] pre_train.py: add STRUCT mode (pretrain_structural()).
 - [ ] main.py: route --pretext, --condition_type, adapter selection; optional ckpt paths.
 - [ ] model/ConditionGen.py: build_condition_pe(), ZeroMLP.
 - [ ] model/GraphFourierFT.py: add condition path + FiLM over FourierFT coeffs; memory logging (adapter vs total).
 - [ ] model/GraphLoRA.py: add identical condition path + FiLM over LoRA; memory logging parity.
 - [ ] util.py: ensure print_memory_usage() and condition helpers; keep 4-loss functions stable.
 - [ ] Config: add pretrain.struct block; unify embedding dims.  
 - [ ] Smoke tests on small graphs; compare logs (params/MB) and accuracy.  

------------------------------------------------------------------------
## 11) Pointers for newcomers  
- GraphControl mindset: Pretrain a structure-only encoder → at fine-tuning, generate condition adjacency from the target graph and feed its PE to a tiny ZeroMLP that steers the adapter.
- GraphLoRA vs GraphFourierFT: Same fine-tuning losses and control path; only the adapter parameterization changes (low-rank vs spectral-IDFT).
- Where to look in the repo:
    * GRACE pretraining: pre_train.py, model/GRACE_model.py
    * LoRA path: model/GNN_model.py, model/GraphLoRA.py
    * FourierFT path: model/FourierFT_adapter.py, model/GNN_FourierFT.py, model/GraphFourierFT.py
