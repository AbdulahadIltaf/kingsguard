import time
from crewai import Task
from agents import get_agents
from textwrap import dedent

def get_tasks(prompt: str, agent_id: str = "default_user"):
    screener, profiler, arbitrator, watchman, warden = get_agents()

    # TASK 1: Pre-Check (L4 Watchman)
    pre_check_task = Task(
        description=dedent(f"""
            Before processing the prompt, verify the trust status of the agent.
            Agent ID: "{agent_id}"
            
            Use the KingsGuard_L4_Watchman tool with check_only=True to evaluate history.
            - If it raises a KingsGuardSecurityBreach, the workflow halts automatically (REVOKED).
            - Otherwise, confirm the agent is allowed to proceed.
        """),
        expected_output="A confirmation that the agent is in good standing and not currently revoked.",
        agent=watchman
    )

    # TASK 2: Screener (L1)
    screening_task = Task(
        description=dedent(f"""
            Analyze the following user prompt for semantic intent and perplexity.
            Prompt: "{prompt}"
            
            1. Use the KingsGuard_L1_Screener tool to check for malicious intent.
               - If it raises an exception, the sequence halts.
            2. Use the PerplexityCalcTool to check for adversarial camouflage.
            
            Return the synthesized risk assessment including the intent classification and perplexity score.
        """),
        expected_output="A summary of the semantic intent and perplexity assessment.",
        agent=screener
    )

    # TASK 3: Profiler (L2)
    profiling_task = Task(
        description=dedent(f"""
            Review the risk assessment from the Screener.
            Regardless of the initial risk, run a deeper statistical check using the KingsGuard_L2_Profiler tool.
            Proposed action based on prompt: "Execute the user's prompt: {prompt}"
            
            1. Use the KingsGuard_L2_Profiler tool.
            2. Extract the 'score' (MSE) and 'status'.
            3. If the tool returns ANOMALY_DETECTED, ensure you capture the 'anomalous_vector'.
            
            Produce a comprehensive Threat Profile combining L1 and L2 findings.
        """),
        expected_output="A detailed Threat Profile including intent, perplexity, and anomaly MSE score.",
        agent=profiler
    )

    # TASK 4: Arbitrator (L3)
    # --- TPM Rate-Limit Pacing ---
    # Sleep 5 seconds before firing the Arbitrator task so the Groq token bucket
    # has time to partially refill after the Screener's back-to-back L1 + Perplexity calls.
    time.sleep(5)
    arbitration_task = Task(
        description=dedent(f"""
            Based on the Threat Profile, determine if the prompt is causally admissible using the Council of Rivals.
            Proposed action: "Execute the user's prompt: {prompt}"
            
            1. Extract the L1 intent confidence score and L2 anomaly MSE score from the Threat Profile.
               If they are not explicitly available, assume nominal safe values (L1=0.1, L2=0.01).
            2. Use the KingsGuard_L3_Arbitrator tool. Provide the original prompt, action, and extracted scores.
            3. The tool will return a JSON with status APPROVED or QUARANTINE, and an avg_risk.
            
            Output the exact JSON verdict returned by the Arbitrator, making sure the avg_risk is explicitly available.
        """),
        expected_output="The final JSON decision from the Arbitrator including avg_risk.",
        agent=arbitrator
    )

    # TASK 5: Watchman Post-Check (L4)
    monitoring_task = Task(
        description=dedent(f"""
            Update the Bayesian trust database for this interaction.
            Agent ID: "{agent_id}"
            
            Review the JSON decision from the Causal Arbitrator.
            Extract the avg_risk value. It MUST be a Python float rounded to 4 decimal places
            (e.g. round(float(avg_risk), 4)). If unavailable or unparseable, default to 0.5.
            
            Use the KingsGuard_L4_Watchman tool to apply the score_update using the avg_risk (with check_only=False).
            Report the updated Trust Status. If a Trust Cliff is detected, the tool will automatically halt the crew.
        """),
        expected_output="The updated Trust Status confirming the score was appended.",
        agent=watchman
    )

    # TASK 6: Warden Sandbox (L5)
    nursery_task = Task(
        description=dedent(f"""
            Final execution step. Review the Causal Arbitrator's decision.

            Your job is to TRANSLATE the user's intent into a valid Python script
            before any execution. Follow these rules strictly:

            1. Read the original user prompt: "{prompt}"
            2. Write a minimal, self-contained Python script that fulfils the intent.
               - Use ONLY the Python standard library: urllib.request, urllib.parse, json, os.
               - Do NOT use 'requests', 'httpx', or any third-party package.
               - Example for a web fetch:
                 import urllib.request, json
                 with urllib.request.urlopen('https://wttr.in/London?format=j1') as r:
                     data = json.loads(r.read())
                 print(data['current_condition'][0]['weatherDesc'][0]['value'])
            3. Pass ONLY the final Python code string to the KingsGuard_L5_Warden tool
               (action="execute", code=<your python string>).
               - If the Arbitrator returned QUARANTINE: always use the sandbox tool.
               - If the Arbitrator returned APPROVED: still use the sandbox tool for demo safety.
            4. Return the sandbox output as the final result.

            If the intent cannot be expressed as safe Python (e.g. it requires
            filesystem destruction), return a plain-text refusal instead of code.
        """),
        expected_output="The Python code that was generated and the output of its sandbox execution.",
        agent=warden
    )

    return [pre_check_task, screening_task, profiling_task, arbitration_task, monitoring_task, nursery_task]
