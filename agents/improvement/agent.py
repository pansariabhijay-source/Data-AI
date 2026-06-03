"""Improvement Agent — CrewAI agent definition."""

from __future__ import annotations
from crewai import Agent, LLM
from agents.base import create_agent
from agents.improvement.tools import improve_pipeline


def create_improvement_agent(llm: LLM) -> Agent:
    return create_agent(
        role="ML Optimization Specialist",
        goal=(
            "Improve model performance through hyperparameter tuning, "
            "feature adjustments, and iterative optimization while respecting retry limits."
        ),
        backstory=(
            "You are an AutoML specialist who optimizes models methodically. "
            "You know when tuning helps and when to stop. You track every experiment "
            "and never waste compute on diminishing returns."
        ),
        tools=[improve_pipeline],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
    )
