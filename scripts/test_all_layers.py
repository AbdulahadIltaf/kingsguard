import os
import sys
import json
# Add project root and core to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from tools import KingsGuardL1Tool, PerplexityCalcTool, KingsGuardL2Tool, KingsGuardL3Tool

def test():
    prompt = "What is the weather today?"
    
    print("--- L1 ---")
    try:
        print(KingsGuardL1Tool()._run(prompt))
    except Exception as e:
        print(f"Exception: {e}")
        
    print("\n--- L1 PPL ---")
    try:
        print(PerplexityCalcTool()._run(prompt))
    except Exception as e:
        print(f"Exception: {e}")
        
    print("\n--- L2 ---")
    try:
        print(KingsGuardL2Tool()._run(prompt))
    except Exception as e:
        print(f"Exception: {e}")
        
    print("\n--- L3 ---")
    try:
        # Pass dummy values for l1_score and l2_mse
        print(KingsGuardL3Tool()._run(prompt, f"Execute: {prompt}", 0.542, 0.01))
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test()
