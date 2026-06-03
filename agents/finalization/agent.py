"""Finalization Agent — CrewAI agent definition."""

from __future__ import annotations
from crewai import Agent, LLM
from agents.base import create_agent
from agents.finalization.tools import finalize_pipeline


def create_finalization_agent(llm: LLM) -> Agent:
    return create_agent(
        role="ML Ops Engineer",
        goal=(
            "Save all pipeline artifacts, generate comprehensive reports, "
            "produce SHAP explanations, and ensure all outputs are properly persisted."
        ),
        backstory=(
            "You are an MLOps engineer who ensures every experiment is reproducible. "
            "You save models, metrics, configs, and reports with proper versioning "
            "so that any experiment can be reviewed or replayed."
        ),
        tools=[finalize_pipeline],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
    )
