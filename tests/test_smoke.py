"""Smoke tests for ml-automation-cv — validate plugin layout invariants."""
from __future__ import annotations

import json
import re
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def test_plugin_manifest_parses_and_has_required_fields():
    """Test that manifest exists, parses JSON, and has required fields."""
    manifest_path = PLUGIN_ROOT / ".cortex-plugin" / "plugin.json"

    # Check file exists
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"

    # Check JSON parses
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check required fields
    required_fields = {"name", "version", "description"}
    for field in required_fields:
        assert field in manifest, f"Missing required field: {field}"

    # Check name starts with spark-
    assert manifest["name"].startswith("spark-"), f"Plugin name must start with 'spark-', got: {manifest['name']}"


def test_agents_md_lists_all_agents_and_skills():
    """Test that AGENTS.md lists all agents and skills from their directories."""
    agents_md_path = PLUGIN_ROOT / "AGENTS.md"

    # Check AGENTS.md exists
    assert agents_md_path.exists(), f"AGENTS.md not found at {agents_md_path}"

    # Read AGENTS.md
    with open(agents_md_path) as f:
        agents_content = f.read()

    # Check all agents/*.md stems appear in AGENTS.md
    agents_dir = PLUGIN_ROOT / "agents"
    for agent_file in agents_dir.glob("*.md"):
        agent_name = agent_file.stem
        # Use word boundary regex to find exact matches
        pattern = rf"\b{re.escape(agent_name)}\b"
        assert re.search(pattern, agents_content), f"Agent '{agent_name}' not found in AGENTS.md"

    # Check all skills/*/ directory names appear in AGENTS.md
    skills_dir = PLUGIN_ROOT / "skills"
    if skills_dir.exists():
        for skill_subdir in skills_dir.iterdir():
            if skill_subdir.is_dir():
                skill_name = skill_subdir.name
                # Use word boundary regex to find exact matches
                pattern = rf"\b{re.escape(skill_name)}\b"
                assert re.search(pattern, agents_content), f"Skill '{skill_name}' not found in AGENTS.md"
