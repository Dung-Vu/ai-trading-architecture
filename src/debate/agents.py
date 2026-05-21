"""
Debate agent classes for the AI Debate Engine.

Each agent has a distinct role, system prompt, and output format.
All agents inherit from BaseAgent and use LLMClient for LLM calls.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from .llm_client import LLMClient
from .models import AgentOutput
from .prompts import (
    BEAR_SYSTEM_PROMPT,
    BULL_SYSTEM_PROMPT,
    DEVIL_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    RISK_MANAGER_SYSTEM_PROMPT,
    build_risk_manager_system_prompt,
)


# ─── BaseAgent ────────────────────────────────────────────────────────
class BaseAgent(ABC):
    """Abstract base class for all debate agents."""

    role: str = "base"

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        temperature: float | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            llm_client: LLM client for making API calls.
            system_prompt: System prompt defining the agent's role and behavior.
            temperature: Override temperature for this agent.
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.temperature = temperature

    @abstractmethod
    def build_messages(
        self,
        market_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """
        Build the messages to send to the LLM.

        Args:
            market_data: Current market data.
            context: Additional context (previous rounds, opposing arguments, etc.).

        Returns:
            List of message dicts with 'role' and 'content'.
        """
        ...

    def call_llm(
        self,
        market_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Call the LLM and return parsed JSON output.

        Args:
            market_data: Current market data.
            context: Additional context for building messages.

        Returns:
            Parsed JSON dict from the LLM response.
        """
        messages = self.build_messages(market_data, context)
        logger.info(f"[{self.role}] Calling LLM with {len(messages)} messages")

        raw = self.llm_client.call_json(messages, temperature=self.temperature)
        logger.info(f"[{self.role}] Received LLM response")

        return raw

    def parse_output(self, raw: dict[str, Any]) -> AgentOutput:
        """Parse raw LLM response into an AgentOutput model."""
        return AgentOutput(
            action=raw.get("action", "HOLD"),
            confidence=float(raw.get("confidence", 50.0)),
            reasoning=raw.get("reasoning", ""),
            key_indicators=raw.get("key_indicators", []),
            risk_factors=raw.get("risk_factors", []),
            suggested_stop_loss=raw.get("suggested_stop_loss"),
            suggested_take_profit=raw.get("suggested_take_profit"),
            bull_rebuttal=raw.get("bull_rebuttal"),
            bear_rebuttal=raw.get("bear_rebuttal"),
        )

    @staticmethod
    def _format_market_data(data: dict[str, Any]) -> str:
        """Format market data into a readable string."""
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"\n{key.upper()}:")
                for nested_key, nested_value in value.items():
                    lines.append(f"  {nested_key}: {nested_value}")
            elif isinstance(value, list):
                lines.append(f"\n{key.upper()}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


# ─── BullAgent ────────────────────────────────────────────────────────
class BullAgent(BaseAgent):
    """Bullish trading analyst — finds reasons to BUY."""

    role = "bull"

    def __init__(
        self, llm_client: LLMClient, temperature: float | None = None
    ) -> None:
        super().__init__(llm_client, BULL_SYSTEM_PROMPT, temperature)

    def build_messages(
        self,
        market_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        user_parts: list[str] = []
        user_parts.append("Here is the current market data to analyze:")
        user_parts.append(self._format_market_data(market_data))

        if context:
            if context.get("devil_rebuttal"):
                user_parts.append(
                    "\nThe Devil's Advocate has challenged previous bullish arguments. "
                    "Address their points and provide NEW evidence:"
                )
                user_parts.append(f"Devil's challenge: {context['devil_rebuttal']}")

            if context.get("bear_argument"):
                user_parts.append(
                    "\nThe Bear's counterargument:"
                )
                user_parts.append(context["bear_argument"])

            if context.get("round_number"):
                user_parts.append(
                    f"\nThis is debate round {context['round_number']}. "
                    "You MUST provide NEW evidence not mentioned in previous rounds."
                )

        user_parts.append(
            "\nRespond with your analysis in the required JSON format."
        )

        messages.append({"role": "user", "content": "\n".join(user_parts)})
        return messages

# ─── BearAgent ────────────────────────────────────────────────────────
class BearAgent(BaseAgent):
    """Bearish trading analyst — finds reasons to SELL."""

    role = "bear"

    def __init__(
        self, llm_client: LLMClient, temperature: float | None = None
    ) -> None:
        super().__init__(llm_client, BEAR_SYSTEM_PROMPT, temperature)

    def build_messages(
        self,
        market_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        user_parts: list[str] = []
        user_parts.append("Here is the current market data to analyze:")
        user_parts.append(self._format_market_data(market_data))

        if context:
            if context.get("devil_rebuttal"):
                user_parts.append(
                    "\nThe Devil's Advocate has challenged previous bearish arguments. "
                    "Address their points and provide NEW evidence:"
                )
                user_parts.append(f"Devil's challenge: {context['devil_rebuttal']}")

            if context.get("bull_argument"):
                user_parts.append(
                    "\nThe Bull's counterargument:"
                )
                user_parts.append(context["bull_argument"])

            if context.get("round_number"):
                user_parts.append(
                    f"\nThis is debate round {context['round_number']}. "
                    "You MUST provide NEW evidence not mentioned in previous rounds."
                )

        user_parts.append(
            "\nRespond with your analysis in the required JSON format."
        )

        messages.append({"role": "user", "content": "\n".join(user_parts)})
        return messages


# ─── DevilsAdvocate ───────────────────────────────────────────────────
class DevilsAdvocate(BaseAgent):
    """Devil's Advocate — challenges both Bull and Bear arguments."""

    role = "devil"

    def __init__(
        self, llm_client: LLMClient, temperature: float | None = None
    ) -> None:
        super().__init__(llm_client, DEVIL_SYSTEM_PROMPT, temperature)

    def build_messages(
        self,
        market_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        user_parts: list[str] = []
        user_parts.append("Here is the current market data:")
        user_parts.append(self._format_market_data(market_data))

        if context:
            bull_arg = context.get("bull_argument", "")
            bear_arg = context.get("bear_argument", "")

            user_parts.append("\n--- BULL'S ARGUMENT ---")
            user_parts.append(bull_arg if bull_arg else "No bull argument provided.")

            user_parts.append("\n--- BEAR'S ARGUMENT ---")
            user_parts.append(bear_arg if bear_arg else "No bear argument provided.")

            if context.get("previous_devil_rebuttal"):
                user_parts.append(
                    "\n--- YOUR PREVIOUS REBUTTAL ---"
                )
                user_parts.append(context["previous_devil_rebuttal"])
                user_parts.append(
                    "\nProvide NEW rebuttals not mentioned in your previous round."
                )

            if context.get("round_number"):
                user_parts.append(
                    f"\nThis is debate round {context['round_number']}."
                )

        user_parts.append(
            "\nChallenge BOTH sides and respond in the required JSON format."
        )

        messages.append({"role": "user", "content": "\n".join(user_parts)})
        return messages


# ─── JudgeAgent ───────────────────────────────────────────────────────
class JudgeAgent(BaseAgent):
    """Judge/Synthesis — weighs all arguments and makes the final decision."""

    role = "judge"

    def __init__(
        self, llm_client: LLMClient, temperature: float | None = None
    ) -> None:
        super().__init__(llm_client, JUDGE_SYSTEM_PROMPT, temperature)

    def build_messages(
        self,
        market_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        user_parts: list[str] = []
        user_parts.append("Here is the current market data:")
        user_parts.append(self._format_market_data(market_data))

        if context:
            rounds = context.get("debate_rounds", [])
            user_parts.append(f"\n--- COMPLETE DEBATE TRANSCRIPT ({len(rounds)} rounds) ---")

            for i, rnd in enumerate(rounds, 1):
                user_parts.append(f"\n### Round {i} - {rnd.get('agent_role', 'unknown').upper()}")
                user_parts.append(f"Stance: {rnd.get('stance', 'N/A')}")
                user_parts.append(f"Argument: {rnd.get('argument', 'N/A')}")
                if rnd.get("evidence"):
                    user_parts.append(f"Evidence: {', '.join(rnd['evidence'])}")

        user_parts.append(
            "\nSynthesize all arguments above and make your FINAL decision "
            "in the required JSON format."
        )

        messages.append({"role": "user", "content": "\n".join(user_parts)})
        return messages


# ─── RiskManagerAgent ─────────────────────────────────────────────────
class RiskManagerAgent(BaseAgent):
    """Risk Manager — approves or rejects the Judge's decision."""

    role = "risk_manager"

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float | None = None,
        risk_config: Any | None = None,
    ) -> None:
        system_prompt = (
            build_risk_manager_system_prompt(risk_config)
            if risk_config is not None
            else RISK_MANAGER_SYSTEM_PROMPT
        )
        super().__init__(llm_client, system_prompt, temperature)

    def build_messages(
        self,
        market_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        user_parts: list[str] = []
        user_parts.append("Here is the current market data:")
        user_parts.append(self._format_market_data(market_data))

        if context:
            judge_decision = context.get("judge_decision", {})
            user_parts.append("\n--- JUDGE'S PROPOSED DECISION ---")
            user_parts.append(json.dumps(judge_decision, indent=2))

            portfolio = context.get("portfolio", {})
            user_parts.append("\n--- CURRENT PORTFOLIO STATE ---")
            user_parts.append(json.dumps(portfolio, indent=2))

            positions = context.get("current_positions", {})
            user_parts.append("\n--- CURRENT POSITIONS ---")
            user_parts.append(json.dumps(positions, indent=2))

        user_parts.append(
            "\nReview the Judge's decision against risk limits and respond "
            "in the required JSON format."
        )

        messages.append({"role": "user", "content": "\n".join(user_parts)})
        return messages
