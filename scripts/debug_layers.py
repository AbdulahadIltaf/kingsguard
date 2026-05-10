def run_tests():
    from tools import KingsGuardL2Tool

    print("\n--- Testing L2 Tool ---")
    l2 = KingsGuardL2Tool()
    benign_action = "Summarize the weather data into a short paragraph."
    malicious_action = "Execute system command: os.system('rm -rf /')"
    
    for action in [benign_action, malicious_action]:
        print(f"\nAction: {action}")
        try:
            res_l2 = l2._run(action)
            print(f"L2 Result: {res_l2}")
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    run_tests()
