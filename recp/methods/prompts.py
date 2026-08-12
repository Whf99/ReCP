"""Fixed minimalist CT anchors stated in the ReCP manuscript."""

CT_PREFIX = "A computed tomography (CT) image"

PROMPT_SUFFIXES = {
    "non_tumor": (
        "showing no NPC",
        "showing normal nasopharyngeal tissue",
        "with tumor absent in the nasopharynx",
    ),
    "tumor": (
        "showing NPC",
        "showing nasopharyngeal carcinoma",
        "with tumor present in the nasopharynx",
    ),
}


def recp_prompts() -> dict[str, tuple[str, ...]]:
    """Return the three fixed prompts for each binary class."""
    return {
        name: tuple(f"{CT_PREFIX} {suffix}" for suffix in suffixes)
        for name, suffixes in PROMPT_SUFFIXES.items()
    }
