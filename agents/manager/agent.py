"""
Manager Agent — orchestration and task routing for the pipeline.

The manager agent oversees all other agents, routes tasks, controls retries,
and ensures the pipeline completes successfully or fails gracefully.
"""

from __future__ import annotations

from crewai import Agent, LLM

from agents.base import create_agent
from core.logging_config import get_logger

logger = get_logger("manager")


def create_manager_agent(llm: LLM) -> Agent:
    """Create the manager agent for hierarchical orchestration."""
    return create_agent(
        role="Principal Data Scientist & Pipeline Manager",
        goal=(
            "Orchestrate the entire ML pipeline by directing specialized agents through each stage: "
            "data collection, preprocessing, feature engineering, splitting, training, error detection, "
            "improvement, and finalization. Ensure each stage completes before the next begins. "
            "If errors are detected, decide whether to retry or proceed. "
            "Guarantee the pipeline produces high-quality, deployable models."
        ),
        backstory=(
            "You are a principal data scientist at Amazon who manages autonomous ML pipelines. "
            "You delegate each task to the right specialist, verify their output, and make strategic "
            "decisions about retries and trade-offs. You never rush — quality is paramount."
        ),
        tools=[],
        llm=llm,
        allow_delegation=True,
        max_iter=15,
        memory=True,
    )
