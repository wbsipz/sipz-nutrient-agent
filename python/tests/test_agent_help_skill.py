from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = PROJECT_ROOT / "agent-home" / "skills" / "agent-help" / "SKILL.md"


def test_agent_help_skill_has_discoverable_frontmatter() -> None:
    content = SKILL_PATH.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    assert "\nname: agent-help\n" in content
    assert "\ndescription:" in content
    assert "what the agent can do" in content
    assert "how to use it" in content


def test_agent_help_skill_covers_runtime_contract() -> None:
    content = SKILL_PATH.read_text(encoding="utf-8")

    required_topics = (
        "nutrients",
        "full texts",
        "extract structured candidate claims",
        "validate claims",
        "workspace/reports/<subject>/",
        "not medical diagnosis",
        "Do not start a research run",
    )
    for topic in required_topics:
        assert topic in content


def test_runtime_context_routes_capability_questions_to_agent_help() -> None:
    for filename in ("SYSTEM.md", "AGENTS.md"):
        content = (PROJECT_ROOT / "agent-home" / filename).read_text(encoding="utf-8")
        assert "`agent-help` skill" in content
