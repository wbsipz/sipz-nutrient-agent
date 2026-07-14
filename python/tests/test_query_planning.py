from sipz_agent.core.query_planning import plan_retrieval_queries


class FakeQueryPlanner:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def complete_json(self, prompt, adapter):
        assert "Requested nutrient/bioactive: anethole" in prompt
        if self.error:
            raise self.error
        return adapter.validate_python(self.response)


def test_query_planner_expands_niche_bioactive_and_keeps_original_query() -> None:
    plan = plan_retrieval_queries(
        "anethole",
        FakeQueryPlanner(
            {
                "canonical_name": "trans-anethole",
                "is_niche": True,
                "specific_synonyms": ["trans-anethole", "p-propenylanisole"],
                "source_terms": ["anise", "fennel"],
                "recommended_queries": [
                    '"trans-anethole" human health oral',
                    "p-propenylanisole human health",
                    "anise trans-anethole clinical",
                ],
                "rationale": "The compound is commonly indexed under its trans isomer.",
            }
        ),
    )

    assert plan.is_niche is True
    assert plan.canonical_name == "trans-anethole"
    assert plan.recommended_queries == [
        "anethole health human",
        '"trans-anethole" human health oral',
        "p-propenylanisole human health",
        "anise trans-anethole clinical",
    ]


def test_query_planner_adds_consolidated_query_for_missing_synonyms() -> None:
    plan = plan_retrieval_queries(
        "anethole",
        FakeQueryPlanner(
            {
                "canonical_name": "anethole",
                "is_niche": True,
                "specific_synonyms": [
                    "trans-anethole",
                    "p-propenylanisole",
                    "1-methoxy-4-(1-propenyl)benzene",
                ],
                "source_terms": ["anise"],
                "recommended_queries": ["trans-anethole human health"],
                "rationale": "Niche compound.",
            }
        ),
    )

    combined_queries = " ".join(plan.recommended_queries)
    assert "trans-anethole" in combined_queries
    assert "p-propenylanisole" in combined_queries
    assert "1-methoxy-4-(1-propenyl)benzene" in combined_queries
    assert len(plan.recommended_queries) <= 6


def test_query_planner_falls_back_when_llm_is_unavailable() -> None:
    plan = plan_retrieval_queries(
        "anethole",
        FakeQueryPlanner(error=RuntimeError("provider unavailable")),
    )

    assert plan.recommended_queries == ["anethole health human"]
    assert "provider unavailable" in plan.rationale
