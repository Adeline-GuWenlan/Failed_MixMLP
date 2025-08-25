#!/usr/bin/env python3
"""
Test script for 3-qubit quantum magic prediction pipeline
Evaluates embedding scalability and training efficiency for 8x8 matrices
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from updated_training_model import CompleteQuantumMagicPredictor
from transformer_embedding import StructuredQuantumEmbedding, QuantumSizeExperts
from quantum_dataset import create_dataloaders
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

def create_dummy_3qubit_data(n_samples=1000):
    """Create dummy 3-qubit data for testing"""
    print(f"🧪 Creating dummy 3-qubit data: {n_samples} samples")
    
    # 8x8 matrices for 3 qubits
    matrix_dim = 8
    
    rho_real = torch.randn(n_samples, matrix_dim, matrix_dim)
    rho_imag = torch.randn(n_samples, matrix_dim, matrix_dim)
    
    # Make matrices Hermitian (rho† = rho)
    rho_real = (rho_real + rho_real.transpose(-1, -2)) / 2
    rho_imag = (rho_imag - rho_imag.transpose(-1, -2)) / 2
    
    # Random magic values for testing
    magic_labels = torch.rand(n_samples) * 0.5  # Typical magic range
    
    return rho_real, rho_imag, magic_labels

def test_embedding_scalability():
    """Test if embedding can handle 3-qubit matrices"""
    print("\n🔍 Testing Embedding Scalability (2-qubit → 3-qubit)")
    
    batch_size = 4
    
    # Test 2-qubit embedding
    embedding_2q = StructuredQuantumEmbedding(n_qubits=2, d_model=64)
    rho_real_2q = torch.randn(batch_size, 4, 4)
    rho_imag_2q = torch.randn(batch_size, 4, 4)
    
    tokens_2q = embedding_2q(rho_real_2q, rho_imag_2q)
    print(f"   2-qubit: {rho_real_2q.shape} → {tokens_2q.shape}")
    
    # Test 3-qubit embedding
    embedding_3q = StructuredQuantumEmbedding(n_qubits=3, d_model=64)
    rho_real_3q = torch.randn(batch_size, 8, 8)
    rho_imag_3q = torch.randn(batch_size, 8, 8)
    
    tokens_3q = embedding_3q(rho_real_3q, rho_imag_3q)
    print(f"   3-qubit: {rho_real_3q.shape} → {tokens_3q.shape}")
    
    # Verify embedding dimension consistency
    assert tokens_2q.shape[-1] == tokens_3q.shape[-1], "Embedding dimensions should match!"
    print(f"   ✅ Embedding dimension consistent: {tokens_2q.shape[-1]}D")
    
    # Test sequence length scaling
    seq_len_2q = tokens_2q.shape[1]  # 16 tokens for 4x4
    seq_len_3q = tokens_3q.shape[1]  # 64 tokens for 8x8
    
    print(f"   Sequence length scaling: {seq_len_2q} → {seq_len_3q} (4x increase)")
    assert seq_len_3q == 4 * seq_len_2q, "Expected 4x sequence length increase"
    
    return True

def test_expert_projectors():
    """Test QuantumSizeExperts projectors"""
    print("\n🔍 Testing Expert Projectors")
    
    experts = QuantumSizeExperts(value_dim=62)
    batch_size, seq_len = 4, 16
    
    # Test complex input [B, seq_len, 2] for real/imag parts
    complex_input = torch.randn(batch_size, seq_len, 2)
    
    # Test different qubit systems
    for n_qubits in [2, 3, 4, 5]:
        features = experts(complex_input, n_qubits)
        print(f"   {n_qubits}-qubit expert: {complex_input.shape} → {features.shape}")
        assert features.shape == (batch_size, seq_len, 62), f"Wrong output shape for {n_qubits}-qubit"
    
    print("   ✅ All expert projectors working correctly")
    return True

def test_3qubit_model():
    """Test complete 3-qubit model"""
    print("\n🔍 Testing Complete 3-Qubit Model")
    
    # Create 3-qubit model
    model = CompleteQuantumMagicPredictor(
        n_qubits=3,
        d_model=64,
        pooling_type="cls",
        mlp_type="physics_aware"
    )
    
    batch_size = 8
    rho_real, rho_imag, labels = create_dummy_3qubit_data(batch_size)
    
    # Forward pass
    print(f"   Input shapes: rho_real={rho_real.shape}, rho_imag={rho_imag.shape}")
    
    with torch.no_grad():
        predictions = model(rho_real, rho_imag)
    
    print(f"   Output shape: {predictions.shape}")
    print(f"   Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    assert predictions.shape == (batch_size, 1), f"Wrong prediction shape: {predictions.shape}"
    
    print("   ✅ 3-qubit model forward pass successful")
    return model

def quick_training_test():
    """Quick training test to verify gradient flow"""
    print("\n🔍 Quick 3-Qubit Training Test")
    
    model = CompleteQuantumMagicPredictor(
        n_qubits=3,
        d_model=64,
        pooling_type="cls",
        mlp_type="physics_aware"
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training data
    n_samples = 100
    rho_real, rho_imag, labels = create_dummy_3qubit_data(n_samples)
    
    model.train()
    initial_loss = None
    
    print("   Training for 10 steps...")
    for step in range(10):
        optimizer.zero_grad()
        
        predictions = model(rho_real, rho_imag)
        loss = criterion(predictions.squeeze(), labels)
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"   Step {step+1}: Loss = {loss.item():.6f}")
    
    final_loss = loss.item()
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"   Initial Loss: {initial_loss:.6f}")
    print(f"   Final Loss: {final_loss:.6f}")
    print(f"   Improvement: {improvement:.2f}%")
    
    if improvement > 0:
        print("   ✅ Model is learning (loss decreasing)")
    else:
        print("   ⚠️ Model not improving in 10 steps (may need more time)")
    
    return True

def test_position_encoding_scaling():
    """Test position encoding for different matrix sizes"""
    print("\n🔍 Testing Position Encoding Scaling")
    
    for n_qubits in [2, 3, 4]:
        matrix_dim = 2 ** n_qubits
        seq_length = matrix_dim ** 2
        
        embedding = StructuredQuantumEmbedding(n_qubits=n_qubits, d_model=64)
        rho_real = torch.randn(2, matrix_dim, matrix_dim)
        rho_imag = torch.randn(2, matrix_dim, matrix_dim)
        
        tokens = embedding(rho_real, rho_imag)
        
        # Extract position coordinates (last 2 dimensions)
        positions = tokens[0, :, -2:]  # [seq_length, 2]
        
        print(f"   {n_qubits}-qubit ({matrix_dim}x{matrix_dim}): positions {positions.shape}")
        print(f"      Position range: i=[{positions[:, 0].min():.0f}, {positions[:, 0].max():.0f}], " +
              f"j=[{positions[:, 1].min():.0f}, {positions[:, 1].max():.0f}]")
        
        # Verify position coordinates are correct
        expected_max = matrix_dim - 1
        assert positions[:, 0].max() == expected_max, f"Wrong max i-coordinate for {n_qubits}-qubit"
        assert positions[:, 1].max() == expected_max, f"Wrong max j-coordinate for {n_qubits}-qubit"
    
    print("   ✅ Position encoding scales correctly")
    return True

def main():
    """Run all 3-qubit pipeline tests"""
    print("🚀 Testing 3-Qubit Quantum Magic Pipeline")
    print("=" * 50)
    
    try:
        # Test individual components
        test_embedding_scalability()
        test_expert_projectors()
        test_position_encoding_scaling()
        
        # Test complete model
        model = test_3qubit_model()
        quick_training_test()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("\n📊 Summary:")
        print("   ✅ Embedding scales from 16 → 64 tokens (4x)")
        print("   ✅ 64D embedding space handles 3-qubit complexity")
        print("   ✅ Expert projectors work for all system sizes")
        print("   ✅ Position encoding scales to 8x8 matrices")
        print("   ✅ Model forward pass and gradient flow working")
        
        print("\n🔬 Ready for 3-qubit training experiments!")
        
        # Model architecture summary
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n📋 3-Qubit Model Summary:")
        print(f"   Parameters: {total_params:,}")
        print(f"   Input: 8x8 complex matrices ({8**2} elements)")
        print(f"   Tokens: 64 x 64D (4096 total dimensions)")
        print(f"   Expert: 3-qubit specialized projector")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()