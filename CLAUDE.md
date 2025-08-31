# CLAUDE.md - Heterophilic-aware GraphLoRA Implementation Guide

## CURRENT IMPLEMENTATION STATUS (2025-08-31)

### ✅ COMPLETED TASKS:
1. **Simple Dataset-based Heterophily Detection** - Implemented in `model/GraphLoRA.py`:
   - Datasets ['Chameleon', 'Squirrel', 'Actor'] classified as heterophilic
   - Other datasets classified as homophilic
   - No complex homophily detector needed for initial implementation

2. **Efficient 2-hop Regularization** - Implemented in `model/GraphLoRA.py`:
   - Added `compute_efficient_2hop()` function for exact 2-hop adjacency calculation
   - Formula: `(A² - A) + I` where A² removes 1-hop connections, +I adds self-loops
   - Modified `calculate_reg_loss()` to accept `use_2hop` parameter
   - Automatic switching: heterophilic datasets use 2-hop, homophilic use 1-hop

3. **Integration and Testing** - Successfully working:
   - Dataset-based switching integrated in training loop
   - PubMed→Squirrel experiment runs successfully with 2-hop regularization
   - PubMed→CiteSeer experiment runs successfully with 1-hop regularization  
   - No CUDA assertion errors or numerical instability

4. **Code Structure** - Clean implementation:
   - Single `calculate_reg_loss()` function handles both 1-hop and 2-hop cases
   - Proper dtype handling (int→float conversion)
   - Minimal code changes to existing GraphLoRA structure

### ⚠️ PENDING TASKS:
1. **Comprehensive Testing** - Need systematic evaluation:
   - Baseline tests on all homophilic datasets (Cora, CiteSeer, PubMed)
   - Tests on all heterophilic datasets (Chameleon, Squirrel, Actor) 
   - Cross-domain transfer experiments
   
2. **Performance Analysis**:
   - Accuracy comparison: original vs. 2-hop regularization
   - Training time and memory usage analysis
   
3. **Code Cleanup**:
   - Remove debug print statements
   - Optimize 2-hop computation if needed
   - Add proper documentation

### 📝 KEY IMPLEMENTATION DETAILS:
- **2-hop Adjacency Formula**: `(A² - A) > 0` with self-loops added
- **Dataset Classification**: Simple name-based detection (no complex homophily computation)
- **Numerical Stability**: Fixed dtype issues, no gradient explosion
- **Backward Compatibility**: Original GraphLoRA functionality preserved for homophilic graphs

## Project Overview  
Implement heterophilic-aware modifications to GraphLoRA to handle both homophilic and heterophilic graphs effectively. The core insight: heterophilic patterns should be treated as features, not bugs.

## Problem Statement (RESOLVED)
Original GraphLoRA failed on heterophilic datasets (Chameleon, Squirrel, Actor) due to:
- ~~SMMD loss assumes neighboring nodes are similar~~ (Still uses original SMMD)
- ~~Contrastive loss treats same-class nodes as positive pairs~~ (Still uses original contrastive)
- ✅ **Structure regularization enforced adjacent node similarity** (FIXED with 2-hop)
- ✅ **GPU runtime errors from gradient explosion in SMMD computation** (RESOLVED)

## NEXT STEPS FOR COMPLETION:

### 1. Comprehensive Evaluation Protocol
**Objective**: Systematically test 2-hop regularization across all datasets

**Test Matrix**:
```
Source Dataset: PubMed (fixed pretrain)
Target Datasets: 
- Homophilic: Cora, CiteSeer, PubMed (self-transfer)
- Heterophilic: Chameleon, Squirrel, Actor

Expected Results:
- Homophilic: Maintain original performance (1-hop regularization)  
- Heterophilic: Improve performance (2-hop regularization)
```

**Commands**:
```bash
# Homophilic baselines  
python main.py --pretrain_dataset PubMed --test_dataset Cora --is_transfer True --seed 42
python main.py --pretrain_dataset PubMed --test_dataset CiteSeer --is_transfer True --seed 42
python main.py --pretrain_dataset PubMed --test_dataset PubMed --is_transfer True --seed 42

# Heterophilic experiments
python main.py --pretrain_dataset PubMed --test_dataset Chameleon --is_transfer True --seed 42  
python main.py --pretrain_dataset PubMed --test_dataset Squirrel --is_transfer True --seed 42
python main.py --pretrain_dataset PubMed --test_dataset Actor --is_transfer True --seed 42
```

### 2. Performance Analysis & Documentation
- Create results table with accuracy comparisons
- Monitor training time and memory usage
- Document any remaining issues or limitations

### 3. Code Quality & Cleanup  
- Remove debug prints and temporary code
- Add proper docstrings and comments
- Optimize 2-hop computation if needed

## ORIGINAL DESIGN DOCUMENTATION (Reference Only)
*Note: The following sections document the original comprehensive design. Current implementation uses a simpler, working approach.*

### ~~Task 1: Implement Robust Homophily Detector~~ (SKIPPED - Using Dataset Names)
**Status**: Not implemented - using simple dataset name classification instead
```python
def adaptive_distance_weight(ppr_weight, homophily_ratio):
    """
    Dynamic weight adjustment based on homophily level.
    
    Homophilic (h > 0.6): Use original PPR
    Heterophilic (h < 0.4): Use inverse PPR = 1/(1+PPR)
    Mixed (0.4 ≤ h ≤ 0.6): Smooth transition
    """
    if homophily_ratio > 0.6:
        return ppr_weight
    elif homophily_ratio < 0.4:
        return 1.0 / (1.0 + ppr_weight)
    else:
        # Smooth transition between modes
        alpha = (homophily_ratio - 0.4) / 0.2
        return alpha * ppr_weight + (1-alpha) * (1/(1+ppr_weight))
```

### Task 3: Implement Efficient 2-hop Regularization
**Location**: Modify `model/GraphLoRA.py` - `structure_regularization()` function

```python
def adaptive_structure_regularization(adj_matrix, predictions, homophily_ratio):
    """
    Homophilic: Original 1-hop similarity
    Heterophilic: 2-hop similarity (enemy of enemy principle)
    
    Efficiency optimizations:
    - Sparse matrix multiplication for 2-hop
    - Top-k neighbors selection to prevent explosion
    - Cache 2-hop computation across epochs
    """
    if homophily_ratio > 0.6:
        # Original GraphLoRA logic
        return original_structure_reg(adj_matrix, predictions)
    else:
        # 2-hop regularization
        adj_2hop = compute_efficient_2hop(adj_matrix)
        return compute_2hop_similarity(predictions, adj_2hop)

def compute_efficient_2hop(adj_matrix, top_k=10):
    """
    Efficient 2-hop computation with memory constraints.
    Uses sparse matrix multiplication and keeps only top-k connections.
    """
    adj_2hop = adj_matrix @ adj_matrix
    adj_2hop.setdiag(0)  # Remove self-loops
    
    # Keep only top-k to prevent memory explosion
    k = min(top_k, adj_2hop.shape[0] // 10)
    return keep_topk_sparse(adj_2hop, k)
```

### Task 4: Integrate Components in Main Training Script
**Location**: Modify `main.py` (the main fine-tuning script)

```python
# At initialization (after loading config)
from model.homophily_detector import RobustHomophilyDetector
homophily_detector = RobustHomophilyDetector(k_hops=3, n_samples=100)

# Before training (after loading dataset)
homophily_ratio = homophily_detector.detect(graph, labels_train)
print(f"Detected homophily ratio: {homophily_ratio:.3f}")

# Determine dataset type
if homophily_ratio > 0.6:
    mode = "homophilic"
elif homophily_ratio < 0.4:
    mode = "heterophilic"
else:
    mode = "mixed"

# Pass homophily_ratio to GraphLoRA model
model = GraphLoRA(
    input_dim=input_dim,
    output_dim=output_dim,
    homophily_ratio=homophily_ratio,  # New parameter
    **other_params
)

# In training loop (inside main.py's train function)
for epoch in range(num_epochs):
    # Forward pass (unchanged)
    
    # The adaptive losses are now computed inside GraphLoRA.forward()
    # based on the homophily_ratio passed during initialization
    loss = model(features, adj, labels, train_mask)
    
    # Continue with existing optimization
```

## Testing Protocol

### 1. Baseline Comparison
```bash
# Test on homophilic datasets (should maintain performance)
python main.py --config config.yaml --dataset Cora
python main.py --config config.yaml --dataset CiteSeer
python main.py --config config.yaml --dataset PubMed

# Test on heterophilic datasets (should see improvement)
python main.py --config config.yaml --dataset Chameleon
python main.py --config config.yaml --dataset Squirrel
python main.py --config config.yaml --dataset Actor
```

### 2. Ablation Study
Update `config.yaml` to include ablation settings:
```yaml
ablation:
  homophily_detect_only: false
  inverse_weight_only: false
  2hop_reg_only: false
  full_model: true
```

Then run:
```bash
# Test each component individually
python main.py --config config.yaml --dataset Squirrel --ablation homophily_detect_only
python main.py --config config.yaml --dataset Squirrel --ablation inverse_weight_only
python main.py --config config.yaml --dataset Squirrel --ablation 2hop_reg_only
python main.py --config config.yaml --dataset Squirrel --ablation full_model
```

### 3. Performance Metrics
- Track gradient stability (no explosion)
- Measure accuracy improvement on heterophilic datasets
- Monitor computational overhead (< 20% increase acceptable)
- Verify homophilic performance maintenance (< 2% degradation)

## Key Implementation Notes

1. **Sparse Matrix Operations**: Always use scipy.sparse for adjacency matrices to handle large graphs efficiently.

2. **Caching Strategy**: Cache expensive computations (2-hop neighbors, eigendecomposition) when graph structure is static.

3. **Gradient Monitoring**: Add gradient clipping and monitoring to prevent explosion:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

4. **Memory Management**: For large graphs, implement batch-wise 2-hop computation:
```python
def batch_2hop_computation(adj_matrix, batch_size=1000):
    n_nodes = adj_matrix.shape[0]
    adj_2hop_blocks = []
    for i in range(0, n_nodes, batch_size):
        block = adj_matrix[i:i+batch_size] @ adj_matrix
        adj_2hop_blocks.append(block)
    return scipy.sparse.vstack(adj_2hop_blocks)
```

## Expected Challenges & Solutions

### Challenge 1: 2-hop Neighbor Explosion
**Solution**: Implement top-k filtering and approximate 2-hop computation using random sampling for very large graphs.

### Challenge 2: Homophily Estimation Variance
**Solution**: Use multiple random walks and average results. For small graphs, use full graph statistics instead of sampling.

### Challenge 3: Sparse Graph Instability
**Solution**: Apply edge density correction factor when edge_density < 0.01.

## Success Criteria
- ✅ No GPU runtime errors on heterophilic datasets
- ✅ 15-20% accuracy improvement on Chameleon/Squirrel/Actor
- ✅ < 2% performance degradation on Cora/CiteSeer/PubMed
- ✅ Training time increase < 20%
- ✅ Memory usage increase < 30%

## Code Style Guidelines
- Follow existing GraphLoRA code structure
- Add comprehensive docstrings for new functions
- Include type hints for better clarity
- Log important metrics (homophily ratio, mode selection)
- Implement unit tests for new components

## Dependencies to Add
```python
# In requirements.txt
scipy>=1.7.0  # For sparse matrix operations
networkx>=2.6  # For graph analysis (optional)
```

## File Structure After Implementation
```
GraphLoRA/
├── config.yaml (modified: add heterophilic settings)
├── config2.yaml (unchanged)
├── datasets/
├── main.py (modified: integration of homophily detection)
├── pre_train.py (unchanged)
├── model/
│   ├── GNN_model.py (unchanged)
│   ├── GRACE_model.py (unchanged)
│   ├── GraphLoRA.py (modified: SMMD loss, structure reg)
│   └── homophily_detector.py (new)
├── util.py (potentially modified for helper functions)
├── README.md (update with heterophilic capabilities)
└── CLAUDE.md (this file)
```

## Configuration Updates
**Location**: Modify `config.yaml` to include heterophilic settings

```yaml
# Add to existing config.yaml
heterophilic:
  enable_adaptive: true
  homophily_threshold_high: 0.6
  homophily_threshold_low: 0.4
  k_hop_sampling: 3
  n_samples: 100
  enable_2hop_reg: true
  max_2hop_neighbors: 10
  
# For experiments
ablation:
  mode: "full"  # Options: "baseline", "homophily_only", "inverse_only", "2hop_only", "full"
```

## Next Steps
1. Implement RobustHomophilyDetector class
2. Modify SMMD loss computation
3. Implement 2-hop regularization
4. Run baseline tests
5. Conduct ablation studies
6. Document results and iterate

Remember: The goal is to make GraphLoRA work on both homophilic and heterophilic graphs without sacrificing performance on either type.