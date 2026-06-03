"""Error Detection Agent — CrewAI agent definition."""

from __future__ import annotations
from crewai import Agent, LLM
from agents.base import create_agent
from agents.error_detection.tools import detect_errors


def create_error_detection_agent(llm: LLM) -> Agent:
    return create_agent(
        role="ML Quality Auditor",
        goal=(
            "Audit the entire pipeline for errors, anomalies, and quality issues. "
            "Detect overfitting, underfitting, data leakage, class imbalance, "
            "and recommend actionable fixes."
        ),
        backstory=(
            "You are a senior ML auditor who has reviewed hundreds of production pipelines. "
            "You have a keen eye for subtle issues like data leakage and overfitting that "
            "less experienced engineers miss."
        ),
        tools=[detect_errors],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
    )
