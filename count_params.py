import os
import sys
import torch
import xgboost as xgb
import numpy as np

# Ensure project root is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("🦇 Analyzing Model Complexity...")

try:
    # 1. LSTM Analysis
    print("\n--- LSTM Model (PyTorch) ---")
    from model_core import BatmanLSTM, DEVICE
    
    # Define architecture based on config defaults in model_core.py
    # Input size is 44 based on current data pipeline
    INPUT_SIZE = 44 
    HIDDEN1 = 80
    HIDDEN2 = 40
    DENSE = 64
    
    model = BatmanLSTM(INPUT_SIZE, HIDDEN1, HIDDEN2, dense_units=DENSE).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Architecture: LSTM({HIDDEN1}) -> Attn -> LSTM({HIDDEN2}) -> Dense({DENSE}) -> Out(1)")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 2. XGBoost Analysis
    print("\n--- XGBoost Model (Gradient Boosting) ---")
    xgb_path = os.path.join(BASE_DIR, "batman_xgb.json")
    if os.path.exists(xgb_path):
        bst = xgb.Booster()
        bst.load_model(xgb_path)
        
        # XGBoost doesn't have "weights" in the same way, but has trees/nodes
        dump = bst.get_dump()
        num_trees = len(dump)
        print(f"Model Status: ACTIVE (Phase 3 Pivot)")
        print(f"Number of Trees: {num_trees}")
        
        # Estimate total nodes/leaves
        total_nodes = 0
        for tree in dump:
            total_nodes += tree.count('\n') # Rough estimate of lines/nodes
        print(f"Total Decision Nodes (approx): {total_nodes:,}")
        
    else:
        print("Model Status: INACTIVE (File not found)")

except Exception as e:
    print(f"❌ Analysis Failed: {e}")
    import traceback
    traceback.print_exc()
