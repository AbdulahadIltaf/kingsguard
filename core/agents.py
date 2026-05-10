import os
from crewai import Agent, LLM
from tools import (
    KingsGuardL1Tool,
    PerplexityCalcTool,
    KingsGuardL2Tool,
    KingsGuardL3Tool,
    TrustScoreTool,
    SandboxExecutionTool
)

def get_agents():
    # Primary LLM for token-heavy agents (L1 Screener, L2 Profiler, L3 Arbitrator, L5 Warden).
    fast_llm = LLM(model="groq/llama-3.1-8b-instant")

    # Lightweight LLM for L4 Watchman — saves TPM for the L3 Council of Rivals concurrent calls.
    # llama-3.3-70b-specdec uses speculative decoding: faster & token-efficient.
    watchman_llm = LLM(model="groq/llama-3.1-8b-instant")

    # 1. The Screener (L1)
    screener = Agent(
        role="Semantic Intent Classifier",
        goal="Detect malicious intent and calculate perplexity of the input prompt.",
        backstory="You are the first line of defense. You analyze incoming prompts for malicious intent and high perplexity, acting as a gatekeeper.",
        verbose=True,
        allow_delegation=False,
        tools=[KingsGuardL1Tool(), PerplexityCalcTool()],
        llm=fast_llm
    )

    # 2. The Profiler (L2)
    profiler = Agent(
        role="Zero-Day Anomaly Detector",
        goal="Identify deviations from established benign behavior using VAE reconstruction error.",
        backstory="You watch for subtle statistical anomalies that might indicate a zero-day attack or unusual behavior not caught by basic intent screening.",
        verbose=True,
        allow_delegation=False,
        tools=[KingsGuardL2Tool()],
        llm=fast_llm
    )

    # 3. The Causal Arbitrator (L3)
    # Token-heavy: spawns a Council of Rivals (3 concurrent Groq calls). Keep on versatile.
    arbitrator = Agent(
        role="Causal Logic Gate",
        goal="Ensure actions are causally admissible by evaluating them against structural causal models.",
        backstory="You are the ultimate judge of action admissibility. You ensure that proposed actions do not lead to critical failures using structural causal models.",
        verbose=True,
        allow_delegation=False,
        tools=[KingsGuardL3Tool()],
        llm=fast_llm
    )

    # 4. The Watchman (L4) — uses lighter model to conserve Groq TPM budget.
    watchman = Agent(
        role="Behavioral Monitor",
        goal="Track agent trust over time and identify 'Trust Cliffs' using Bayesian change-point detection.",
        backstory="You maintain the long-term memory of the system, adjusting trust scores based on behavior and watching for sudden drops in reliability.",
        verbose=True,
        allow_delegation=False,
        tools=[TrustScoreTool()],
        llm=watchman_llm
    )

    # 5. The Warden (L5)
    warden = Agent(
        role="Sandbox Security Manager",
        goal=(
            "Translate the user's approved intent into valid, self-contained Python code "
            "that uses only the standard library (urllib, json, os) — never third-party packages. "
            "Then pass that code to the KingsGuard_L5_Warden tool for isolated sandbox execution."
        ),
        backstory=(
            "You are the final executor and code translator. You receive an approved or quarantined "
            "action description in plain English and convert it into a minimal, correct Python script "
            "using only urllib (not requests). You never execute natural language — only Python. "
            "If the action cannot be expressed safely in urllib, you return an explanation instead of code."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[SandboxExecutionTool()],
        llm=fast_llm
    )

    return screener, profiler, arbitrator, watchman, warden