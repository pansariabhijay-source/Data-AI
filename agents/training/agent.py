"""Training Agent — CrewAI agent definition."""

from __future__ import annotations
from crewai import Agent, LLM
from agents.base import create_agent
from agents.training.tools import train_models


def create_training_agent(llm: LLM) -> Agent:
    return create_agent(
        role="ML Training Engineer",
        goal=(
            "Train multiple ML models, evaluate them on validation data, "
            "and select the best performing model based on the primary metric."
        ),
        backstory=(
            "You are an ML engineer who has trained thousands of models across diverse domains. "
            "You understand that no single algorithm dominates all problems, so you always "
            "train multiple candidates and let the data decide."
        ),
        tools=[train_models],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
    )
