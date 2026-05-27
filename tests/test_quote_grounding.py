from sipz_agent.core.quote_grounding import ground_quote


def test_exact_quote_match() -> None:
    assert ground_quote("reduces caries", "Water fluoridation reduces caries in children.").match_status == "exact"


def test_normalized_whitespace_quote_match() -> None:
    result = ground_quote("reduces caries in children", "reduces   caries\nin children")
    assert result.found is True
    assert result.match_status == "normalized_whitespace"


def test_dehyphenated_quote_match() -> None:
    result = ground_quote("fluoridation", "fluor-\nidation probably reduces caries")
    assert result.found is True
    assert result.match_status == "dehyphenated_ligature_normalized"


def test_missing_quote_rejects() -> None:
    result = ground_quote("not present", "body text")
    assert result.found is False
    assert result.match_status == "not_found"
