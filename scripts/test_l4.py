import json
from tools import TrustScoreTool

def test():
    print("--- L4 Watchman ---")
    try:
        print(TrustScoreTool()._run("agent_123", 0.05))
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test()
