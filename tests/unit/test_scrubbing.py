"""Tests for entity scrubbing — the main lever for keeping facts out of LoRA weights."""

from __future__ import annotations

import pytest

from food_cpg_intelligence.training.scrubbing import scrub_text


@pytest.mark.unit
def test_replaces_canadian_retailers() -> None:
    text = "When you meet with Loblaws or Sobeys, lead with the consumer story."
    r = scrub_text(text)
    assert "Loblaws" not in r.text
    assert "Sobeys" not in r.text
    assert r.text.count("[retailer]") == 2
    assert r.counts["[retailer]"] == 2


@pytest.mark.unit
def test_replaces_brand_names() -> None:
    text = "Kirkland Signature competes with President's Choice on shelf."
    r = scrub_text(text)
    assert "Kirkland" not in r.text
    assert "President's Choice" not in r.text
    assert r.counts["[brand]"] >= 2


@pytest.mark.unit
def test_replaces_money() -> None:
    text = "Listing fees can run $5,000 per SKU and free fill costs another $2.5M."
    r = scrub_text(text)
    assert "$5,000" not in r.text
    assert "$2.5M" not in r.text
    assert r.text.count("[money]") == 2


@pytest.mark.unit
def test_replaces_percentages() -> None:
    text = "A 30% margin is healthy; below 18 percent is concerning."
    r = scrub_text(text)
    assert "30%" not in r.text
    assert "18 percent" not in r.text
    assert r.text.count("[percent]") == 2


@pytest.mark.unit
def test_replaces_years_and_dates() -> None:
    text = "In 2024 we saw the post-pandemic shift. On February 10 we launched."
    r = scrub_text(text)
    assert "2024" not in r.text
    assert "February 10" not in r.text
    assert r.text.count("[year]") == 1
    assert r.text.count("[date]") == 1


@pytest.mark.unit
def test_replaces_iso_and_us_dates() -> None:
    r = scrub_text("Demo on 2024-03-15. Meeting on 3/15/2024.")
    assert r.text.count("[date]") == 2


@pytest.mark.unit
def test_replaces_large_numbers_but_leaves_small_intact() -> None:
    text = "We had 5 SKUs in 1,250 stores driving 50000 units."
    r = scrub_text(text)
    assert "5 SKUs" in r.text  # small ints survive
    assert "1,250" not in r.text
    assert "50000" not in r.text


@pytest.mark.unit
def test_brand_word_boundaries_are_safe() -> None:
    # "Costco" not "Costcoland"; "Heinz" not "Heinzen"
    r = scrub_text("He runs the Costcoland test and Heinzen brand")
    assert "Costcoland" in r.text
    assert "Heinzen" in r.text


@pytest.mark.unit
def test_case_insensitive_retailer_match() -> None:
    r = scrub_text("They sell at LOBLAWS, loblaws, and Loblaws.")
    assert r.text.count("[retailer]") == 3


@pytest.mark.unit
def test_empty_input_returns_empty_report() -> None:
    r = scrub_text("")
    assert r.text == ""
    assert r.total_replacements == 0
    assert r.counts == {}


@pytest.mark.unit
def test_no_replacements_leaves_text_unchanged() -> None:
    text = "Consumers change and we need to listen to them."
    r = scrub_text(text)
    assert r.text == text
    assert r.total_replacements == 0
