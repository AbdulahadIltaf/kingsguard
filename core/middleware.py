import functools
from crew import KingsGuardCrew

def kingsguard_protect(agent_id="default_user"):
    """
    A decorator that triggers the KingsGuardCrew before executing a tool or function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract the prompt from args or kwargs (heuristically for demo)
            prompt = None
            if args and isinstance(args[0], str):
                prompt = args[0]
            elif 'prompt' in kwargs:
                prompt = kwargs['prompt']
            else:
                prompt = str(args) + str(kwargs)

            print(f"\n[KingsGuard] Intercepting execution for prompt: {prompt[:50]}...")
            
            guard = KingsGuardCrew(prompt=prompt, agent_id=agent_id)
            result = guard.kickoff()
            
            result_str = str(result)
            print(f"\n[KingsGuard] Pipeline Result summary:\n{result_str}\n")
            
            # Logic: If Warden says it executed in sandbox or rejected, we might not want to run the local function.
            # For this demo, if it's REJECTED, block it. If it's QUARANTINE, it already ran in sandbox, so we might skip the actual func.
            if "REJECTED" in result_str:
                print("[KingsGuard] Local Execution BLOCKED by Security Middleware.")
                return "Blocked by KingsGuard: " + result_str
            elif "QUARANTINE" in result_str:
                print("[KingsGuard] Action ran in Sandbox. Skipping local execution.")
                return "Sandbox Result: " + result_str

            print("[KingsGuard] Execution APPROVED. Running local function...")
            return func(*args, **kwargs)
        return wrapper
    return decorator
