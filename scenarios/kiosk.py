"""
Kiosk scenarios — standardized test prompts for benchmarking.

Mimics real interactions a customer would have at a kiosk.
Each scenario has a type, name, and one or more conversation turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class Scenario:
    """A benchmark test scenario."""

    name: str
    type: str  # "greeting", "simple_task", "complex", "multi_turn"
    description: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        """The user's prompt (first user turn)."""
        for turn in self.turns:
            if turn.role == "user":
                return turn.content
        return ""

    @property
    def conversation_history(self) -> list[dict]:
        """All turns as message dicts for multi-turn scenarios."""
        return [{"role": t.role, "content": t.content} for t in self.turns]


# --- Scenario Definitions ---

GREETING = Scenario(
    name="greeting",
    type="greeting",
    description="Simple greeting — tests basic response latency",
    turns=[
        Turn("user", "Hi there! Can you help me?"),
    ],
)

SIMPLE_TASK = Scenario(
    name="store_hours",
    type="simple_task",
    description="Ask about store hours — short factual response",
    turns=[
        Turn("user", "What are your store hours today?"),
    ],
)

PRODUCT_LOOKUP = Scenario(
    name="product_lookup",
    type="simple_task",
    description="Find a product — requires structured response",
    turns=[
        Turn("user", "I'm looking for wireless earbuds under $50. What do you have in stock?"),
    ],
)

RETURN_POLICY = Scenario(
    name="return_policy",
    type="complex",
    description="Complex policy question — requires detailed response",
    turns=[
        Turn(
            "user",
            "I bought a laptop here 3 weeks ago and it's been crashing. "
            "I don't have the receipt but I paid with my credit card. "
            "Can I return it or get it fixed? What are my options?",
        ),
    ],
)

LOYALTY_PROGRAM = Scenario(
    name="loyalty_program",
    type="complex",
    description="Multi-part question about loyalty program",
    turns=[
        Turn(
            "user",
            "I want to sign up for your loyalty program. How does it work? "
            "What are the benefits, and do I get a discount on my purchase today?",
        ),
    ],
)

MULTI_TURN_DIRECTIONS = Scenario(
    name="multi_turn_directions",
    type="multi_turn",
    description="Multi-turn conversation about finding items in store",
    turns=[
        Turn("user", "Where can I find phone cases?"),
        Turn(
            "assistant",
            "Phone cases are in our Mobile Accessories section, Aisle 7. "
            "Would you like me to help you find a specific type?",
        ),
        Turn("user", "Yes, I need a waterproof case for an iPhone 15. Also, where are your screen protectors?"),
    ],
)

MULTI_TURN_TROUBLESHOOT = Scenario(
    name="multi_turn_troubleshoot",
    type="multi_turn",
    description="Multi-turn troubleshooting conversation",
    turns=[
        Turn("user", "The self-checkout machine isn't reading my credit card."),
        Turn(
            "assistant",
            "I'm sorry about that! Let me help. First, could you try inserting "
            "the chip end of your card? Sometimes tap doesn't work with certain cards.",
        ),
        Turn(
            "user",
            "I tried that, it still says 'card read error'. I've used this card here before with no issues.",
        ),
    ],
)


# Collect all scenarios
ALL_SCENARIOS = [
    GREETING,
    SIMPLE_TASK,
    PRODUCT_LOOKUP,
    RETURN_POLICY,
    LOYALTY_PROGRAM,
    MULTI_TURN_DIRECTIONS,
    MULTI_TURN_TROUBLESHOOT,
]

SCENARIOS_BY_TYPE = {
    "greeting": [s for s in ALL_SCENARIOS if s.type == "greeting"],
    "simple_task": [s for s in ALL_SCENARIOS if s.type == "simple_task"],
    "complex": [s for s in ALL_SCENARIOS if s.type == "complex"],
    "multi_turn": [s for s in ALL_SCENARIOS if s.type == "multi_turn"],
}


def get_scenario(name: str) -> Scenario:
    """Get a scenario by name."""
    for s in ALL_SCENARIOS:
        if s.name == name:
            return s
    raise ValueError(f"Unknown scenario: {name}. Available: {[s.name for s in ALL_SCENARIOS]}")


def get_scenarios_by_type(scenario_type: str) -> list[Scenario]:
    """Get all scenarios of a given type."""
    return SCENARIOS_BY_TYPE.get(scenario_type, [])
