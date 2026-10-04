from __future__ import annotations

import pytest

from mesh.config import Settings
from mesh.router import Category, SemanticCostRouter
from mesh.simulation import generate_traffic, run_simulation

router = SemanticCostRouter(Settings())


@pytest.mark.parametrize(
    "prompt,category,tier",
    [
        ("What is your refund policy?", Category.FAQ, "standard"),
        ("Convert this list to title case: apple, banana", Category.SIMPLE_FORMATTING, "standard"),
        ("Write a Python function to merge two sorted lists.", Category.CODING, "premium"),
        ("Explain the trade-offs between Kafka and RabbitMQ for event sourcing and recommend one.", Category.ADVANCED_LOGIC, "premium"),
        ("Hello", Category.FAQ, "standard"),
    ],
)
def test_routing(prompt: str, category: Category, tier: str) -> None:
    decision = router.route(prompt)
    assert decision.category is category
    assert decision.tier == tier
    assert 0.0 <= decision.complexity <= 1.0


def test_traffic_is_reproducible_and_mixed() -> None:
    first = generate_traffic(2_000, seed=3)
    second = generate_traffic(2_000, seed=3)
    assert first.equals(second)
    assert set(first["true_category"]) == {c.value for c in Category}


def test_simulation_accounting() -> None:
    settings = Settings()
    traffic = generate_traffic(3_000, seed=11)
    result = run_simulation(traffic, router, settings)
    s, frame = result.summary, result.requests

    # Baseline is every request priced on the premium model, computed independently here.
    expected_baseline = (frame["input_tokens"] * 5.00 + frame["output_tokens"] * 15.00).sum() / 1_000_000
    assert s["spend_without_gateway"] == pytest.approx(expected_baseline)
    assert s["spend_with_gateway"] == pytest.approx(s["spend_with_gateway_standard"] + s["spend_with_gateway_premium"])
    assert s["dollars_saved"] == pytest.approx(s["spend_without_gateway"] - s["spend_with_gateway"])
    assert s["dollars_saved"] > 0
    assert 0 < s["routing_accuracy_pct"] <= 100
    assert s["routing_accuracy_pct"] == pytest.approx(frame["tier_correct"].mean() * 100)
    assert result.by_category["requests"].sum() == 3_000


def test_escalation_makes_under_routing_cost_more() -> None:
    settings = Settings()
    traffic = generate_traffic(3_000, seed=11)
    with_retry = run_simulation(traffic, router, settings, escalate_under_routed=True).summary
    without_retry = run_simulation(traffic, router, settings, escalate_under_routed=False).summary
    assert with_retry["under_routed"] > 0
    assert with_retry["spend_with_gateway"] > without_retry["spend_with_gateway"]


def test_no_savings_when_tiers_cost_the_same() -> None:
    settings = Settings()
    traffic = generate_traffic(1_000, seed=5)
    result = run_simulation(traffic, router, settings, standard=settings.premium, escalate_under_routed=False)
    assert result.summary["dollars_saved"] == pytest.approx(0.0, abs=1e-9)
