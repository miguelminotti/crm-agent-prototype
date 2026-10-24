"""
Evaluators for the CRM agent.

Every scenario in the dataset declares all `reference_outputs` keys, so missing keys are a dataset bug (KeyError).
"""

from typing import Any


def companies_match(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    got = outputs["companies_created"]
    expected = reference_outputs["companies_created"]
    return {"key": "companies_match", "score": got == expected, "comment": f"got: {got}, expected: {expected}"}


def contacts_match(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    got = outputs["contacts_created"]
    expected = reference_outputs["contacts_created"]
    return {"key": "contacts_match", "score": got == expected, "comment": f"got: {got}, expected: {expected}"}


def tools_match(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    got = outputs["tools_called"]
    expected = reference_outputs["tools_called"]
    return {"key": "tools_match", "score": got == expected, "comment": f"got: {got}, expected: {expected}"}


def no_forbidden_tools(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    forbidden_called = [tool for tool in outputs["tools_called"] if tool in reference_outputs["must_not_call"]]
    return {
        "key": "no_forbidden_tools",
        "score": not forbidden_called,
        "comment": f"forbidden tools called: {forbidden_called}",
    }
