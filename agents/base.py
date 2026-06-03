"""
Base agent factory — common LLM and agent creation logic.

All agents share the same LLM config and a standard creation pattern.
This factory centralizes that to avoid duplication across 8+ agent modules.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from crewai import Agent, LLM

from core.config import LLMConfig
from core.logging_config import get_logger

logger = get_logger("agent_factory")


def create_llm(config: Optional[LLMConfig] = None) -> LLM:
    """Create the LLM instance used by all agents.

    Reads CEREBRAS_API_KEY from environment. Falls back to config defaults.
    """
    if config is None:
        config = LLMConfig()

    model = os.environ.get("LLM_MODEL", config.model)
    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    temperature = float(os.environ.get("LLM_TEMPERATURE", str(config.temperature)))

    if not api_key:
        logger.warning("CEREBRAS_API_KEY not set — LLM calls will fail")

    return LLM(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=config.max_tokens,
    )


def create_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: list[Any],
    llm: LLM,
    *,
    verbose: bool = True,
    allow_delegation: bool = False,
    max_iter: int = 5,
    memory: bool = True,
) -> Agent:
    """Create a CrewAI agent with standardized settings.

    Args:
        role: The agent's role description.
        goal: What the agent is trying to achieve.
        backstory: Context about the agent's expertise.
        tools: List of CrewAI tools available to the agent.
        llm: The LLM instance to use.
        verbose: Whether to log agent reasoning.
        allow_delegation: Whether agent can delegate to others.
        max_iter: Maximum reasoning iterations.
        memory: Whether to enable agent memory.

    Returns:
        Configured CrewAI Agent.
    """
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools,
        llm=llm,
        verbose=verbose,
        allow_delegation=allow_delegation,
        max_iter=max_iter,
        memory=memory,
    )
    logger.info(f"Agent created: {role}")
    return agent
