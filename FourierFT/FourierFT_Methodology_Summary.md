# FourierFT Methodology Summary

FourierFT: Parameter-Efficient Fine-Tuning with Discrete Fourier
Transform (ICML 2024)

------------------------------------------------------------------------

## Phase 1. FourierFT Adapter Construction

**1-1. Input**\
- Pretrained weight matrix ( W_0
`\in `{=tex}`\mathbb{R}`{=tex}\^{d\_{in} `\times `{=tex}d\_{out}} ) from
a frozen base layer (e.g., GATConv in GraphLoRA).

**1-2. Process**\
1. Treat weight update (`\Delta `{=tex}W) as a matrix in spatial
domain.\
2. Randomly select (n) spectral entries (coordinates in frequency
domain).\
3. Initialize a trainable coefficient vector (c
`\in `{=tex}`\mathbb{R}`{=tex}\^n).\
4. Construct sparse spectral matrix (F
`\in `{=tex}`\mathbb{R}`{=tex}\^{d\_{in} `\times `{=tex}d\_{out}}):\
- (F\_{i,j} = c_l) if ((i,j)) is a chosen entry, else (0).\
5. Compute Inverse Discrete Fourier Transform (IDFT):\
\[ `\Delta `{=tex}W =
`\alpha `{=tex}`\cdot `{=tex}`\text{IDFT}`{=tex}(F) \]\
where (`\alpha`{=tex}) is a scaling hyperparameter.

**1-3. Output**\
- Updated weight:\
\[ W = W_0 + `\Delta `{=tex}W \]\
- Can be merged directly with frozen base weights during fine-tuning.

------------------------------------------------------------------------

## Phase 2. Forward Pass with FourierFT

**2-1. Input**\
- Node embedding (or feature vector) (x).\
- Graph structure (edge index).

**2-2. Process**\
1. Compute base layer output: ( h\_{base} = W_0 x ).\
2. Compute FourierFT update: ( h\_{ft} = `\Delta `{=tex}W x ).\
3. Combine outputs:\
\[ h = h\_{base} + h\_{ft} \]

**2-3. Output**\
- Node embedding updated with FourierFT-adapted weights.

------------------------------------------------------------------------

## Phase 3. Integration into Transfer Learning

**3-1. Input**\
- Pretrained GNN encoder (frozen).\
- FourierFT-adapted branch (trainable).\
- Downstream dataset (features, edges, labels).

**3-2. Process**\
1. Use FourierFT adapters in place of LoRA for each fine-tuned layer.\
2. Apply classification head, projector, and auxiliary losses (same as
GraphLoRA).\
- Classification loss (L\_{cls})\
- Structure-aware MMD loss (L\_{smmd})\
- Contrastive loss (L\_{contrastive})\
- Structural regularization (L\_{struct_reg})

**3-3. Output**\
- Final node embeddings adapted to target graph.\
- Prediction logits and overall transfer loss.

------------------------------------------------------------------------

## Phase 4. Efficiency Logging

**4-1. Input**\
- FourierFT adapter parameters (c) and entry indices (E).\
- Entire trainable parameter set (adapter + projector + classifier).

**4-2. Process**\
1. Count number of trainable parameters in FourierFT branch.\
2. Count total number of trainable parameters in full model.\
3. Convert to memory (MB) by multiplying with parameter size.\
4. Print logs each epoch for direct comparison with LoRA.

**4-3. Output**\
- Example log:
`Adapter params (FourierFT): 48K (0.19 MB)   Total trainable params: 1.3M (5.2 MB)`

------------------------------------------------------------------------

📌 핵심 요약:\
FourierFT는 LoRA의 low-rank factorization 대신, **weight 변화 ΔW를
frequency domain의 sparse coefficient로 parameterize**하고, **IDFT로
spatial domain weight를 복원**한다. 결과적으로 훨씬 적은 파라미터로도
comparable한 표현력을 유지할 수 있다.
