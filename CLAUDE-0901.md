# 📑 GraphFourierFT Program Design Document (with Scaffold)

------------------------------------------------------------------------

## 1. Objective

Replace the **LoRA components** in GraphLoRA with **FourierFT
adapters**, while preserving the original training pipeline
(pretraining, transfer learning, four-part loss). Add logging utilities
to directly compare **parameter efficiency and memory usage** of LoRA vs
FourierFT.

Please refer to the file ./FourierFT/FourierFT.pdf or ./FourierFT/FourierFT_Methodology_Summary.pdf for help with implementation.
------------------------------------------------------------------------

## 2. Key Requirements

1.  **Loss Consistency**: Keep

    $$\mathcal{L} = \lambda_1 L_{cls} + \lambda_2 L_{smmd} + \lambda_3 L_{contrastive} + \lambda_4 L_{struct\_reg}$$
    exactly as in GraphLoRA.
2.  **Parameter-Efficient Tuning**: Replace only the **LoRA branch**
    with FourierFT. Projector, LogReg, and backbone GNN remain as in
    GraphLoRA.
3.  **Memory Logging**:
    -   Log **total learnable parameters**.\
    -   Log **adapter-only parameters** (LoRA or FourierFT).\
    -   Print memory (MB) for both, to allow efficiency comparison.\
4.  **Compatibility**: Keep interface signatures consistent with
    GraphLoRA for seamless integration.

------------------------------------------------------------------------

## 3. Module Overview

### 3.1 FourierFT Adapter (new)

**File:** `model/FourierFT_adapter.py`\
**Class:** `FourierFTAdapter`\
**Signature:**

``` python
class FourierFTAdapter(nn.Module):
    def __init__(self, d_in, d_out, n=1000, alpha=1.0, base_layer=None):
        ...
    def forward(self, x, edge_index):
        ...
```

**Responsibilities**: - Store **randomly chosen spectral entries E**
(frozen across all layers).\
- Maintain **trainable coefficients c (size n)**.\
- Build spectral matrix F with coefficients placed at entries E.\
- Compute ΔW = IDFT(F) (use `torch.fft.ifft2`) and merge with base layer
weights.\
- Return updated embeddings.

**Notes for engineer**: - You do not need to optimize IDFT; use
PyTorch's FFT utilities.\
- Ensure ΔW has the same shape as the base GNN weight matrix.

------------------------------------------------------------------------

### 3.2 GNN with FourierFT

**File:** `model/GNN_FourierFT.py`\
**Class:** `GNNFourierFT`\
**Signature:**

``` python
class GNNFourierFT(nn.Module):
    def __init__(self, gnn, gnn_type, gnn_layer_num, d_in, d_out, n, alpha):
        ...
    def forward(self, x, edge_index):
        ...
```

**Responsibilities**: - Wrap pretrained GNN (frozen).\
- For each fine-tuned layer (matching LoRA logic), attach a
FourierFTAdapter.\
- Forward pass = base output + FourierFT update.\
- Return embeddings `(emb, emb_base, emb_fourier)` for downstream loss
terms.

------------------------------------------------------------------------

### 3.3 Transfer Function

**File:** `model/GraphFourierFT.py`\
**Function:** `transfer_fourier(args, config, gpu_id, is_reduction)`

**Responsibilities**: - Same structure as `GraphLoRA.transfer()`.\
- Load pretrained GNN.\
- Freeze base parameters.\
- Replace LoRA injection (`GNNLoRA`) with `GNNFourierFT`.\
- Initialize Projector, LogReg, optimizer groups.\
- Compute four losses each epoch, log training/validation/test accuracy.

**Additional Logging**: - Use utility `print_trainable_parameters()`
(already in `util.py`) but extend to: - Print **adapter-only params**
(FourierFT coefficients only).\
- Print **all trainable params** (adapter + projector + classifier).\
- Example log output:
`Adapter params (FourierFT): 64K (0.25 MB)   Total trainable params: 1.2M (4.8 MB)`

------------------------------------------------------------------------

### 3.4 Memory Logging Utility

**File:** `util.py` (extend existing)\
**Function:**

``` python
def print_memory_usage(model, adapter_module_names=[]):
    ...
```

-   Report:
    -   Total trainable parameters (count + MB).\
    -   Adapter-only parameters (count + MB).\
-   Identify adapters via `adapter_module_names` or isinstance check.

------------------------------------------------------------------------

## 4. Workflow

1.  **Pretraining**
    -   No change. Use `pre_train.py` with GRACE.\
2.  **Transfer (FourierFT)**
    -   Replace call to `GraphLoRA.transfer` with
        `GraphFourierFT.transfer_fourier`.\
    -   Attach FourierFT adapters instead of LoRA.\
3.  **Training Loop**
    -   Compute losses: cls, smmd, contrastive, struct_reg.\
    -   Optimize projector + logreg + FourierFT adapters.\
    -   Log parameter efficiency at start and after each epoch.

------------------------------------------------------------------------

## 5. Task Delegation Checklist

**Engineer Deliverables**: 1. **Implement FourierFTAdapter** - Trainable
coefficients, spectral entry initialization, IDFT weight
reconstruction. - Merge ΔW with base weights in forward. 2. **Implement
GNNFourierFT** - Attach FourierFTAdapter modules to frozen GNN layers. -
Ensure outputs compatible with loss computation. 3. **Modify Transfer
Function** - New `transfer_fourier()` function in `GraphFourierFT.py`. -
Preserve 4-loss logic. 4. **Extend Logging** - Implement
`print_memory_usage()`. - Integrate at beginning of training and
per-epoch summary. 5. **Testing** - Run on small dataset (Cora,
CiteSeer). - Verify losses decrease, memory logs show adapter-only
parameters.

------------------------------------------------------------------------

## 6. Expected Outcomes

-   Functional GraphFourierFT pipeline (pretrain + Fourier-based
    transfer).\
-   Logs showing reduced adapter parameter count compared to LoRA.\
-   Accuracy comparable to baseline GraphLoRA (to be validated
    empirically).

------------------------------------------------------------------------

# 📑 Scaffold for GraphFourierFT Integration

## 1. Directory Scaffold (Modifications Plan)

### Root level

-   **`main.py`** → \[Modified\]
    -  Add option to choose `GraphFourierFT.transfer_fourier` instead of existing `GraphLoRA.transfer` call.\
    -   Example: Add `--method {lora, fourier}` argument.
-   **`pre_train.py`** → \[No Change\]
    -   GRACE pretraining logic remains unchanged.
-   **`util.py`** → \[Modified/Added\]
    -   Extend existing `print_trainable_parameters()`.\
    -   \[Added\] `print_memory_usage(model, adapter_module_names=[])`:
        output adapter-only vs total param count/MB

------------------------------------------------------------------------

### `model/` directory

-   **`GNN_model.py`** → \[Modified\]
    -   Keep existing `GNNLoRA` as is, but implement FourierFT version seperately.\
    -   [Added] Or connect `GNNFourierFT` import path here.
-   **`GRACE_model.py`** → \[No Change\]
    -   Self-supervised pretraining logic remains unchanged.
-   **`GraphLoRA.py`** → \[No Change\]
    -   Keep original LoRA-based Transfer.
-   **\[Added\] `GraphFourierFT.py`** (New File)
    -   Function: `transfer_fourier(args, config, gpu_id, is_reduction)`\
    -   Same pipeline as GraphLoRA.transfer but use `GNNFourierFT` instead of `GNNLoRA`.\
    -   Add FourierFT-related logging.
-   **\[Added\] `FourierFT_adapter.py`** (New File)
    -   Class: `FourierFTAdapter(nn.Module)`\
    -   Trainable coefficient c, spectral entry E, restore ΔW through IDFT.\
    -   Return forward result by merging with base_layer.
-   **\[Added\] `GNN_FourierFT.py`** (New File)
    -   Class: `GNNFourierFT(nn.Module)`\
    -   Pretrained GNN(frozen) + FourierFTAdapter branch in parallel.\
    -   Output: `(emb, emb_base, emb_fourier)`

------------------------------------------------------------------------

## 2. Engineer Job Checklist

1.  **Add Option** (`main.py`)
    -   Implement `--method` argument → `'lora'` or `'fourier'`.\
    -   Conditional branching: GraphLoRA vs GraphFourierFT.
2.  **Implement FourierFT Adapter** (`FourierFT_adapter.py`)
    -   `E` spectral entry matrix (Random Sampling, freeze).\
    -   `c` coefficients (trainable).\
    -   `ΔW = torch.fft.ifft2(F).real * α`.\
    -   Merge with base layer output.
3.  **Implement GNNFourierFT** (`GNN_FourierFT.py`)
    -   Pretrained GNN freeze.\
    -   Connect FourierFTAdapter branch in parallel.
4.  **Implement Transfer Loop** (`GraphFourierFT.py`)
    -   Keep loss calculation structure identical.\
    -   Add memory usage logging.
5.  **Extend Utility** (`util.py`)
    -   Add `print_memory_usage()` function: output adapter-only vs total param memory.
6.  **Testing**
    -   Sanity check with Cora / CiteSeer / Chameleon datasets.\
    -   Verify FourierFT adapter params difference compared to LoRA in logs.

------------------------------------------------------------------------

## 3. Expected Logs Example

    [FourierFT] Adapter params: 64K (0.25 MB)
    [FourierFT] Total trainable params: 1.2M (4.8 MB)
    Epoch 1 | train_acc=0.83 | val_acc=0.80 | test_acc=0.78
    ...
