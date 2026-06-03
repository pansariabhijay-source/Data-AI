"""Feature Engineering Agent — CrewAI agent definition."""

from __future__ import annotations
from crewai import Agent, LLM
from agents.base import create_agent
from agents.feature_engineering.tools import engineer_features


def create_feature_engineering_agent(llm: LLM) -> Agent:
    return create_agent(
        role="Feature Engineering Specialist",
        goal=(
            "Transform raw features into ML-ready representations through encoding, "
            "scaling, feature generation, and intelligent feature selection."
        ),
        backstory=(
            "You are a Kaggle grandmaster-level feature engineer who understands that the right "
            "features are more important than the right model. You balance signal preservation "
            "with dimensionality reduction."
        ),
        tools=[engineer_features],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
    )
