# Background:
This project aims to develop a machine learning approach to predict quantum "magic" monotones from density matrices, addressing the computational challenge of traditional geometric methods that scale as O(4^N) and become intractable for systems with more than 5 qubits.

1. Input Data Type: Square Matrix of size, with n represents 
2. Output Data Type: a magic
3. Input Data Feature: 
   1. (features of density matrix)
   2. contain real and imaginary part
   3. imput format: decoded tokens produced by quantum_dataset.py, which intakes the uppertriangular part of the paired matrix real and imaginary(as scalars, without i). 
# Ultimate Goal: 
as for the ultimate goal of extrapolating to 6 qubits: as long as we figure out how the pos embedding work for 2 3 and 4 through visualization (or other approach), we may build similar embedded parts for 6 qubits sys. and then for 6 qubits we will have the 'enbedded token' ready, the same length as the 2-4 ones"

# Current status: 
having trained the model/ whole pipeline for 2 qubits case. 
# Cuurent pipeline: data format and methodology

### 1. Data Processing Chain
```
Raw Density Matrices → Upper Triangular Extraction → Token Encoding → Transformer Processing → Magic Prediction
```

**Key Pipeline Components:**

#### `quantum_dataset.py`
- **Purpose**: Core data loading and preprocessing
- **Input**: Paired `*_real.npy` and `*_imag.npy` files containing density matrix components
- **Format**: [B, D, D] arrays where B=batch size, D=matrix dimension (2^n_qubits)
- **Output**: DataLoader yielding (rho_real, rho_imag, labels) tuples
- **Features**: 
  - Automatic file pairing and validation
  - Train/validation/test splitting with reproducible seeds
  - Optimized loading with configurable batch sizes and workers

#### `transformer_embedding.py` - StructuredQuantumEmbedding
- **Purpose**: Convert matrix elements to transformer-ready tokens
- **Architecture**: 
  ```python
  # Current approach: [quantum_values | position_coordinates]
  d_model = value_features(62D) + position_features(2D)
  ```
- **Key Design**: Separates semantic quantum information from spatial positioning
- **Output**: [B, D², d_model] token sequences

#### `transformer_encoder.py` - PhysicsInformedTransformerEncoder
- **Purpose**: Process quantum tokens with optional physics-aware attention
- **Features**:
  - Multi-head self-attention with optional CLS token support
  - Physics mask capability (currently disabled for stability)
  - Standard transformer blocks with LayerNorm and feedforward layers
- **Configuration**: 4 encoder layers, 8 attention heads by default

#### `updated_training_model.py` - CompleteQuantumMagicPredictor
- **Purpose**: End-to-end model combining all components
- **Architecture Flow**:
  1. Embedding: [B, D, D] → [B, D², d_model]
  2. Optional CLS token addition: [B, D²+1, d_model]
  3. Transformer encoding: Multi-layer processing
  4. Pooling: CLS extraction or physics-aware aggregation
  5. MLP head: Final magic value prediction

#### `mlp.py` - QuantumMagicMLPv2
- **Purpose**: Final prediction head with multiple architecture options
- **Variants**:
  - **Standard**: Progressive dimensionality reduction
  - **Physics-aware**: Separate pathways for eigenvalue/coherence/entanglement features
  - **Attention-enhanced**: Self-attention on intermediate features
  - **Mixture of Experts**: Multiple specialized prediction heads
  - **Asymmetric Ensemble**: Multi-depth pathway combination

### 2. Training Infrastructure

#### `train_single.py`
- **Purpose**: Isolated experiment execution with unique output paths
- **Features**:
  - Automatic GPU selection and memory management
  - Mixed-precision training with PyTorch 2.5.1+ AMP
  - Comprehensive metrics logging (MSE, MAE, R², Spearman, Pearson, acc@tolerance)
  - Early stopping and learning rate scheduling
  - Unique timestamped outputs to prevent conflicts

#### `dispatch.py`
- **Purpose**: Distributed experiment management via tmux sessions
- **Capabilities**:
  - Parallel experiment launching across multiple GPUs
  - Predefined experiment configurations
  - Grid search support for hyperparameter exploration
  - Session monitoring and cleanup utilities

# Current Task:

Multi-System Universal Embedding Concept: The idea of mapping different qubit system all to the same 128-2=126D or 62-2=60D(leave 2 out for the i,j index) embedding space using specialized projectors.
Build a more robust transformer encider + MLP for mixed size matrix(quantum systems composed by didffferent matrix), with num_qubits ranging from 2 to 5. 
- need to figure out what kind of paras is needed to be changed for fitting different dataszie. 
## Breaking down Current Task - Current small Task:
need to try the current model on the 3 qubits case and evaluate:
- whether the current embedding set, good for 2 qubits, can be fit for 3 qubits case as well, or larger embedding sapce is needed? 
  - Is maintaining the current 64-128 embedding sapce possible if we add training epoches for larger size of matrix?
    - for reference, currently for 2 qubits, 200 rounds of training can leads to R**2 > 0.99
- of entry pos embedding, we are simply using (i, j) index as their original representation in the matrix. would the training be as effective if we expand the size to larger matrix size?
- One approach for mixedsize matrix training is having all the input matrix, regardless of their size. 
  - start by building the matrix linear projection layer individually. like this:
    ```py
    class QuantumExperts(nn.Module):
    def __init__(self):
        self.expert_2q = nn.Linear(4, 62)   # "2-qubit specialist"
        self.expert_3q = nn.Linear(6, 62)   # "3-qubit specialist" 
        self.expert_4q = nn.Linear(8, 62)   # "4-qubit specialist"
    
    def forward(self, binary_bits, system_size):
        if system_size == 2:
            return self.expert_2q(binary_bits)      # Only this expert activated
        elif system_size == 3:
            return self.expert_3q(binary_bits)      # Only this expert activated
        # ... each expert specializes in their domain
    ```
    and have mixsized training (after we validate that the structure also works well for the 3 qubits case)

# Quantum Magic Monotone Prediction: Transformer-Based Approach

## Current Status: 2-Qubit System Performance

### Achieved Results
- **R² Score**: >0.99 after 200 training epochs
- **System Size**: 2 qubits (4×4 density matrices)
- **Token Sequence Length**: 16 matrix elements → 16 tokens
- **Embedding Dimension**: 64 (62 learned features + 2 position coordinates)

### Model Configuration
```python
# Current successful configuration
matrix_dim = 4  # 2^2 qubits
d_model = 64
sequence_length = 16  # 4² matrix elements
pooling_type = "cls"  # CLS token aggregation
mlp_type = "physics_aware"  # Physics-informed prediction head
```

### Immediate Objective: 3-Qubit Evaluation
Test current architecture on 3-qubit systems (8×8 matrices) to determine:

1. **Embedding Scalability**: 
   - Can the 64-dimensional embedding space accommodate 3-qubit complexity?
   - Does sequence length expansion (16→64 tokens) require architectural changes?

2. **Position Encoding Effectiveness**:
   - Will simple (i,j) coordinate encoding scale to larger matrices?
   - Do we need more sophisticated positional representations?

3. **Training Efficiency**:
   - Can similar R² performance be achieved with reasonable epoch counts?
   - What architectural modifications improve convergence?

#### Universal Embedding Space Concept
- **Target**: Map all qubit systems to consistent 60-62D feature space
- **Benefit**: Enables mixed-size training and systematic extrapolation
- **Challenge**: Preserving quantum structure across different system sizes

### Research Questions for Next Phase

1. **Architectural Scaling**: What embedding dimensions and sequence lengths are optimal for 3-6 qubit systems?

2. **Physics Preservation**: How can we maintain quantum mechanical constraints during multi-system training?

3. **Extrapolation Capability**: Can a model trained on 2-4 qubit data successfully predict 6-qubit magic values?

4. **Position Encoding**: Should we adopt binary index encoding or tensor product-aware representations for larger systems?

This roadmap positions the project to achieve the ultimate goal of computationally efficient magic monotone prediction for quantum systems beyond current geometric method limitations.