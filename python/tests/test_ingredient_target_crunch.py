from sipz_agent.core.ingredient_target_crunch import build_crunched_run_target_rows


def target(
    name: str,
    *,
    direct: int = 1,
    reuse: int = 0,
    rep_name: str | None = None,
) -> dict[str, str]:
    return {
        "group_id": name.replace(" ", "_"),
        "canonical_search_name": name,
        "direct_row_count": str(direct),
        "reuse_row_count": str(reuse),
        "representative_canonical_beverage_id": f"id-{name.replace(' ', '-')}",
        "representative_canonical_beverage_name": rep_name or name,
        "relationship_examples": "[]",
        "confidence": "0.8",
        "reason": "fixture",
    }


def existing(name: str, covered: list[str]) -> dict[str, str]:
    return {
        "run_target_id": name.replace(" ", "_"),
        "run_target_name": name,
        "representative_canonical_beverage_id": f"existing-{name.replace(' ', '-')}",
        "representative_canonical_beverage_name": name,
        "covered_target_names": str(covered).replace("'", '"'),
        "covered_representative_names": "[]",
        "collapse_reason_code": "fixture",
        "collapse_reason": "fixture",
    }


def by_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["run_target_name"]: row for row in rows}


def test_crunch_maps_added_targets_to_existing_final_targets() -> None:
    rows, summary = build_crunched_run_target_rows(
        canonical_target_rows=[
            target("americano coffee"),
            target("green apple"),
            target("black pepper ground"),
            target("jalapeño pepper"),
            target("tart cherries"),
            target("goji berries"),
            target("new herb"),
        ],
        existing_final_target_rows=[
            existing("coffee", ["coffee", "americano coffee"]),
            existing("apple", ["apple", "green apple"]),
            existing("black pepper", ["black pepper"]),
            existing("chili pepper", ["chili pepper", "jalapeno pepper"]),
            existing("tart cherry", ["tart cherry"]),
            existing("goji berry", ["goji berry"]),
        ],
    )

    mapped = by_name(rows)
    assert mapped["coffee"]["covered_target_names"] == '["americano coffee"]'
    assert mapped["coffee"]["collapse_reason_code"] == "existing_final_covered_target"
    assert mapped["apple"]["covered_target_names"] == '["green apple"]'
    assert mapped["black pepper"]["covered_target_names"] == '["black pepper ground"]'
    assert mapped["chili pepper"]["covered_target_names"] == '["jalapeño pepper"]'
    assert mapped["tart cherry"]["covered_target_names"] == '["tart cherries"]'
    assert mapped["goji berry"]["covered_target_names"] == '["goji berries"]'
    assert mapped["new herb"]["collapse_reason_code"] == "unchanged"
    assert summary["source_target_count"] == 7
    assert summary["run_target_count"] == 7


def test_crunch_collapses_obvious_added_form_variants_to_added_base_target() -> None:
    rows, summary = build_crunched_run_target_rows(
        canonical_target_rows=[
            target("papaya"),
            target("dried papaya"),
            target("papaya puree"),
            target("baby arugula leaves"),
            target("arugula"),
        ],
        existing_final_target_rows=[],
    )

    mapped = by_name(rows)
    assert set(mapped) == {"papaya", "arugula"}
    assert mapped["papaya"]["covered_target_count"] == "3"
    assert mapped["arugula"]["covered_target_count"] == "2"
    assert "added_variant" in mapped["papaya"]["collapse_reason_code"]
    assert summary["run_target_count"] == 2


def test_crunch_preserves_distinct_targets_when_no_base_exists() -> None:
    rows, _ = build_crunched_run_target_rows(
        canonical_target_rows=[
            target("agave inulin"),
            target("algae oil"),
            target("andean pink coarse salt"),
        ],
        existing_final_target_rows=[],
    )

    mapped = by_name(rows)
    assert set(mapped) == {"agave inulin", "algae oil", "andean pink coarse salt"}
    assert all(row["collapse_reason_code"] == "unchanged" for row in rows)


def test_crunch_maps_salt_and_sugar_variants_to_existing_targets() -> None:
    rows, _ = build_crunched_run_target_rows(
        canonical_target_rows=[
            target("smoked sea salt"),
            target("iodized salt"),
            target("light brown cane sugar"),
            target("sugar cane"),
        ],
        existing_final_target_rows=[
            existing("sea salt", ["sea salt"]),
            existing("salt", ["salt"]),
            existing("brown sugar", ["brown sugar"]),
            existing("sugar", ["sugar"]),
        ],
    )

    mapped = by_name(rows)
    assert mapped["sea salt"]["covered_target_names"] == '["smoked sea salt"]'
    assert mapped["salt"]["covered_target_names"] == '["iodized salt"]'
    assert mapped["brown sugar"]["covered_target_names"] == '["light brown cane sugar"]'
    assert mapped["sugar"]["covered_target_names"] == '["sugar cane"]'


def test_crunch_renames_oil_and_broader_food_targets() -> None:
    rows, _ = build_crunched_run_target_rows(
        canonical_target_rows=[
            target("canola"),
            target("canola oil"),
            target("cod liver fish"),
            target("coconut mct"),
            target("romaine lettuce"),
            target("little gem lettuce"),
            target("mixed nuts"),
            target("dill pickle"),
            target("pickle"),
        ],
        existing_final_target_rows=[],
    )

    mapped = by_name(rows)
    assert mapped["canola oil"]["covered_target_count"] == "2"
    assert mapped["cod liver oil"]["covered_target_names"] == '["cod liver fish"]'
    assert mapped["mct oil"]["covered_target_names"] == '["coconut mct"]'
    assert mapped["lettuce"]["covered_target_count"] == "2"
    assert mapped["nuts"]["covered_target_names"] == '["mixed nuts"]'
    assert mapped["pickle"]["covered_target_count"] == "2"


def test_crunch_reuses_existing_targets_for_late_pass_food_forms() -> None:
    rows, _ = build_crunched_run_target_rows(
        canonical_target_rows=[
            target("baby shiitake mushrooms"),
            target("hot pepper"),
            target("tahini"),
        ],
        existing_final_target_rows=[
            existing("chili pepper", ["chili pepper"]),
            existing("sesame seed", ["sesame seed"]),
        ],
    )

    mapped = by_name(rows)
    assert mapped["shiitake mushroom"]["covered_target_names"] == '["baby shiitake mushrooms"]'
    assert mapped["chili pepper"]["covered_target_names"] == '["hot pepper"]'
    assert mapped["sesame seed"]["covered_target_names"] == '["tahini"]'


def test_crunch_marks_low_value_targets_for_exclusion() -> None:
    rows, _ = build_crunched_run_target_rows(
        canonical_target_rows=[target("clamato")],
        existing_final_target_rows=[],
    )

    assert rows[0]["run_target_name"] == "clamato"
    assert rows[0]["collapse_reason_code"] == "excluded_low_value"
