"""Human-readable renderers for the dataset profile and diagnostics artifacts.

The renderers only reformat values that were already measured. They never add a
statistic, never infer one, and never substitute a default for a missing value.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

NA = "not applicable"
MISSING = "_none observed_"


def _text(value: Any) -> str:
    if value is None:
        return NA
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured markdown table."""

    body = list(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_cell(item) for item in row) + " |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(item) for item in value) if value else MISSING
    return _text(value).replace("|", "\\|")


def _example_list(values: Sequence[Any], limit: int = 6) -> str:
    if not values:
        return MISSING
    return " · ".join(f"`{_code(item)}`" for item in list(values)[:limit])


def _code(value: Any) -> str:
    text = str(value)
    return text.replace("`", "'").replace("\n", " ").replace("|", "/")[:160]


def _mapping_table(mapping: Mapping[str, Any], key_header: str, value_header: str, limit: int = 30) -> str:
    if not mapping:
        return MISSING
    items = sorted(mapping.items(), key=lambda item: (-item[1] if isinstance(item[1], int) else 0, item[0]))
    return _table([key_header, value_header], items[:limit])


def _token_pairs(values: Any) -> list[tuple[str, int]]:
    """Accept both ``(token, count)`` tuples and serialized token dictionaries."""

    pairs: list[tuple[str, int]] = []
    for item in values or []:
        if isinstance(item, Mapping):
            pairs.append((str(item.get("token")), int(item.get("count", 0))))
        else:
            token, count = item
            pairs.append((str(token), int(count)))
    return pairs


def _profiled_columns(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the measured column statistics of a file, or an empty list.

    Files outside the read scope keep their column names only, so no statistic
    is ever rendered for data that was not read.
    """

    columns = item.get("columns") or []
    if not columns or not isinstance(columns[0], Mapping):
        return []
    return [column for column in columns if isinstance(column, Mapping)]


def render_markdown(profile: Mapping[str, Any]) -> str:
    """Render the full dataset profile as a reviewable markdown document."""

    parts: list[str] = []
    parts.append("# Phase 1 - Dataset Inspection and Profiling")
    parts.append("")
    parts.append(
        "Every number below was measured from the files listed in this report. "
        "No statistic is estimated, defaulted, or copied from documentation."
    )
    parts.append("")
    parts.append(f"- Generated: `{profile['generated_at']}`")
    parts.append(f"- Dataset root: `{profile['dataset_root']}`")
    parts.append(f"- Read split: `{profile['read_split']}` (test data is not read)")
    parts.append(f"- Rows profiled: {_text(profile['totals']['rows_profiled'])}")
    parts.append("")

    parts.extend(_section_scope(profile))
    parts.extend(_section_files(profile))
    parts.extend(_section_schemas(profile))
    parts.extend(_section_missingness(profile))
    parts.extend(_section_identifiers(profile))
    parts.extend(_section_countries(profile))
    parts.extend(_section_ground_truth(profile))
    parts.extend(_section_cardinality(profile))
    parts.extend(_section_noise(profile))
    parts.extend(_section_variations(profile))
    parts.extend(_section_overlap(profile))
    parts.extend(_section_leakage(profile))
    parts.extend(_section_validation(profile))
    parts.extend(_section_methodology(profile))
    return "\n\n".join(part for part in parts if part) + "\n"


def _section_scope(profile: Mapping[str, Any]) -> list[str]:
    scope = profile["scope"]
    lines = [
        "## 1. Scope and File Discovery",
        "",
        f"Discovered files under `{profile['dataset_root']}`. OS metadata paths "
        f"({_text(len(profile['ignored_paths']))} of them) were excluded by configuration.",
        "",
        _table(
            ["relative path", "filename", "ext", "size MiB", "split", "role", "source", "read in phase 1+2"],
            [
                [
                    item["relative_path"],
                    item["filename"],
                    item["extension"],
                    item["size_mib"],
                    item["split"],
                    item["role"],
                    item["source_index"],
                    "yes" if item["split"] == profile["read_split"] else "no (availability check only)",
                ]
                for item in profile["files"]
            ],
        ),
        "",
        f"Reading is restricted to split `{scope['read_split']}`. "
        f"Files read: {_text(scope['files_read'])}. "
        f"Files discovered but deliberately not read: {_text(scope['files_not_read'])}.",
    ]
    if profile.get("ignored_paths"):
        lines.extend(["", "Ignored metadata paths:", "", _example_list(profile["ignored_paths"], 20)])
    return lines


def _section_files(profile: Mapping[str, Any]) -> list[str]:
    lines = ["## 2. Row Counts and Duplicate Rows", ""]
    rows = [
        [
            item["relative_path"],
            _text(item.get("row_count")),
            _text(item.get("duplicate_row_count")),
            item.get("duplicate_row_method", NA),
        ]
        for item in profile["files"]
        if item.get("row_count") is not None
    ]
    lines.append(_table(["file", "rows", "duplicate rows", "method"], rows))
    return lines


def _section_schemas(profile: Mapping[str, Any]) -> list[str]:
    lines = ["## 3. Exact Schemas", ""]
    for item in profile["files"]:
        columns = _profiled_columns(item)
        lines.append(f"### `{item['relative_path']}`")
        lines.append("")
        lines.append(
            f"Delimiter `{item['delimiter']}` | encoding `{item['encoding']}` | "
            f"{_text(item['column_count'])} column(s) | "
            f"rows: {_text(item.get('row_count')) if item.get('row_count') is not None else 'not read (out of scope)'}"
        )
        col_names = [col["name"] if isinstance(col, Mapping) else str(col) for col in item.get("columns", [])]
        lines.append("Columns: " + ", ".join(f"`{name}`" for name in col_names))
        lines.append("")
        if not columns:
            lines.append(
                "Column-level statistics are not reported for this file because it is "
                "outside the Phase 1 and Phase 2 read scope."
            )
            lines.append("")
            continue
        lines.append(
            _table(
                ["column", "inferred dtype", "nulls", "null %", "distinct", "min len", "max len", "mean len"],
                [
                    [
                        column["name"],
                        column["inferred_dtype"],
                        _text(column["null_count"]),
                        _text(column["null_percentage"]),
                        _text(column["distinct_count"]),
                        _text(column["min_length"]),
                        _text(column["max_length"]),
                        _text(column["mean_length"]),
                    ]
                    for column in columns
                ],
            )
        )
        lines.append("")
    return lines


def _section_missingness(profile: Mapping[str, Any]) -> list[str]:
    lines = ["## 4. Missingness", ""]
    rows = []
    for item in profile["files"]:
        for column in _profiled_columns(item):
            if not column["null_count"]:
                continue
            rows.append(
                [
                    item["relative_path"],
                    column["name"],
                    _text(column["null_count"]),
                    _text(column["null_percentage"]),
                    _text(column["whitespace_only_count"]),
                ]
            )
    lines.append(
        _table(["file", "column", "nulls", "null %", "whitespace-only"], rows)
        if rows
        else "No null values were observed in any column of the training files."
    )
    return lines


def _section_identifiers(profile: Mapping[str, Any]) -> list[str]:
    lines = ["## 5. Identifier and Source Structure", ""]
    rows = []
    for item in profile.get("identifiers", []):
        rows.append(
            [
                item["file"],
                item["column"],
                _text(item["row_count"]),
                _text(item["distinct_count"]),
                item["distinct_method"],
                _text(item["duplicate_id_count"]),
                "yes" if item["id_is_unique"] else "no",
                ", ".join(f"{prefix!r}x{count:,}" for prefix, count in item["prefix_counts"].items()),
            ]
        )
    lines.append(_table(["file", "column", "rows", "distinct", "method", "duplicates", "unique", "prefix counts"], rows))
    lines.append("")
    for item in profile.get("identifiers", []):
        if item["non_numeric_id_count"]:
            lines.append(
                f"- `{item['file']}`: {_text(item['non_numeric_id_count'])} value(s) without a numeric body, "
                f"e.g. {_example_list(item['non_numeric_id_examples'])}"
            )
    return lines


def _section_countries(profile: Mapping[str, Any]) -> list[str]:
    lines = ["## 6. Country Distribution (Open Set)", ""]
    lines.append(
        "Country labels are profiled as observed. No fixed country universe is encoded "
        "anywhere in the pipeline, so an unseen label remains usable."
    )
    lines.append("")
    combined: dict[str, int] = {}
    for item in profile["files"]:
        distribution = item.get("country_distribution")
        if not distribution:
            continue
        lines.append(f"### `{item['relative_path']}`")
        lines.append("")
        lines.append(_mapping_table(distribution, "label", "rows"))
        lines.append("")
        for label, count in distribution.items():
            combined[label] = combined.get(label, 0) + count
    lines.append("### Combined training distribution")
    lines.append("")
    lines.append(_mapping_table(combined, "label", "rows"))
    return lines


def _section_ground_truth(profile: Mapping[str, Any]) -> list[str]:
    truth = profile.get("ground_truth")
    if not truth:
        return []
    lines = [
        "## 7. Training Ground Truth",
        "",
        f"- File: `{truth['file']['relative_path']}`",
        f"- Columns: " + ", ".join(f"`{name}`" for name in truth["file"]["columns"]),
        f"- Source column: `{truth['source_column']}` (S1 representation, prefixes: "
        + ", ".join(f"{key!r}x{value:,}" for key, value in truth["source_id_prefixes"].items())
        + ")",
        f"- Matched column: `{truth['matched_column']}` (comma-separated list, prefixes: "
        + ", ".join(f"{key!r}x{value:,}" for key, value in truth["matched_id_prefixes"].items())
        + ")",
        f"- Rows: {_text(truth['total_rows'])} | distinct S1 ids: {_text(truth['distinct_source1_ids'])} "
        f"| duplicate S1 rows: {_text(truth['duplicate_source1_rows'])}",
        f"- Total matched ids: {_text(truth['total_matched_ids'])}",
        "",
        "### Integrity checks",
        "",
        _table(
            ["check", "value"],
            [
                ["duplicate S1 rows", _text(truth["duplicate_source1_rows"])],
                ["rows with duplicate ids inside one list", _text(truth["duplicate_id_rows"])],
                ["duplicate id occurrences", _text(truth["duplicate_id_occurrences"])],
                ["rows with whitespace-padded ids", _text(truth["whitespace_padded_id_count"])],
                ["rows whose id list is not sorted", _text(truth["unsorted_id_list_rows"])],
                ["rows with a leading/trailing separator", _text(truth["separator_edge_case_rows"])],
                ["rows with an empty-string match cell", _text(truth["empty_string_match_rows"])],
                ["malformed entries detected", _text(truth["malformed_entry_count"])],
            ],
        ),
        "",
    ]
    if truth["malformed_entry_examples"]:
        lines.extend(["Malformed examples:", "", _example_list(truth["malformed_entry_examples"]), ""])

    validation = profile.get("ground_truth_validation")
    if validation:
        lines.extend(["### Matched-ID resolvability against the training sources", ""])
        lines.append(
            _table(
                ["prefix", "declared matches", "resolvable", "unresolvable", "unresolvable %"],
                [
                    [
                        prefix,
                        _text(entry["declared_matches"]),
                        _text(entry.get("resolvable_ids", NA)),
                        _text(entry.get("unresolvable_ids", NA)),
                        _text(
                            round(
                                100.0 * entry.get("unresolvable_ids", 0) / entry["declared_matches"], 6
                            )
                            if entry["declared_matches"]
                            else 0.0
                        ),
                    ]
                    for prefix, entry in sorted(validation["per_prefix"].items())
                ],
            )
        )
        lines.append("")
    return lines


def _section_cardinality(profile: Mapping[str, Any]) -> list[str]:
    truth = profile.get("ground_truth")
    if not truth:
        return []
    total = truth["total_rows"] or 1
    lines = [
        "## 8. Match Cardinality per S1 Entity",
        "",
        _table(
            ["metric", "value", "share of S1 entities"],
            [
                [
                    "S1 entities with zero matches",
                    _text(truth["s1_entities_with_zero_matches"]),
                    _text(round(100.0 * truth["s1_entities_with_zero_matches"] / total, 4)),
                ],
                [
                    "S1 entities with exactly one match",
                    _text(truth["s1_entities_with_one_match"]),
                    _text(round(100.0 * truth["s1_entities_with_one_match"] / total, 4)),
                ],
                [
                    "S1 entities with multiple matches",
                    _text(truth["s1_entities_with_multiple_matches"]),
                    _text(round(100.0 * truth["s1_entities_with_multiple_matches"] / total, 4)),
                ],
                ["average matches per S1", _text(truth["average_matches_per_s1"]), NA],
                ["median matches per S1", _text(truth["median_matches_per_s1"]), NA],
                ["maximum matches per S1", _text(truth["max_matches_per_s1"]), NA],
                ["minimum matches per S1", _text(truth["min_matches_per_s1"]), NA],
            ],
        ),
        "",
        "Match-count distribution:",
        "",
        _mapping_table(truth["match_count_distribution"], "matches per S1", "S1 entities", 40),
        "",
        "Contribution by matched-source prefix:",
        "",
        _mapping_table(truth["match_contribution_by_prefix"], "prefix", "matched ids"),
        "",
    ]
    return lines


def _section_noise(profile: Mapping[str, Any]) -> list[str]:
    samples = profile.get("sample_profiles", {})
    if not samples:
        return []
    lines = [
        "## 9. Observed Noise Patterns",
        "",
        "Counts in this section come from a deterministic leading-row sample. "
        f"Sample size: {_text(profile['sample_rows'])} row(s) per training source file, "
        "taken in file order so the result is reproducible.",
        "",
    ]
    for file_key, columns in samples.items():
        lines.append(f"### `{file_key}`")
        lines.append("")
        for column, data in columns.items():
            lines.append(f"**`{column}`** (sample rows: {_text(data['rows'])})")
            lines.append("")
            counts = data.get("pattern_counts") or {}
            if counts:
                lines.append(
                    _table(
                        ["pattern", "matching values", "share of sample"],
                        [
                            [name, _text(count), _text(round(100.0 * count / data["rows"], 4) if data["rows"] else 0.0)]
                            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                            if count
                        ],
                    )
                )
                lines.append("")
            case = data.get("case_profile") or {}
            if case:
                lines.append(
                    "Capitalization: "
                    + ", ".join(f"{name}={_text(count)}" for name, count in sorted(case.items()))
                )
                lines.append("")
            script = data.get("script_profile") or {}
            if script:
                lines.append(
                    "Script distribution (first 50,000 sampled values): "
                    + ", ".join(f"{name}={_text(count)}" for name, count in script.items())
                )
                lines.append("")
            components = data.get("address_component_count_distribution")
            if components:
                lines.append(
                    "Comma-separated address component counts: "
                    + ", ".join(f"{name} components={_text(count)}" for name, count in components.items())
                )
                lines.append("")
            trailing = data.get("top_trailing_tokens") or {}
            if trailing:
                for column_name, pairs in trailing.items():
                    lines.append(
                        f"Most frequent trailing token in `{column_name}`: "
                        + ", ".join(
                            f"{token!r}x{count:,}" for token, count in _token_pairs(pairs)[:20]
                        )
                    )
                    lines.append("")
            leading = data.get("top_leading_tokens") or {}
            if leading:
                for column_name, pairs in leading.items():
                    lines.append(
                        f"Most frequent leading token in `{column_name}`: "
                        + ", ".join(
                            f"{token!r}x{count:,}" for token, count in _token_pairs(pairs)[:20]
                        )
                    )
                    lines.append("")
            for label, values in sorted((data.get("pattern_examples") or {}).items()):
                if values:
                    lines.append(f"Examples with `{label}`:")
                    lines.append("")
                    lines.append(_example_list(values))
                    lines.append("")
    return lines


def _section_variations(profile: Mapping[str, Any]) -> list[str]:
    report = profile.get("variations")
    if not report:
        return []
    lines = [
        "## 10. Real Variation Examples from Linked Training Pairs",
        "",
        f"Method: `{report['method']}`. Sample size: {_text(report['sample_pairs'])} ground-truth-linked "
        "record pair(s) from the training split. Every example below exists in the data.",
        "",
        _table(
            ["variation", "observed pairs"],
            [[name, _text(count)] for name, count in sorted(report["variation_counts"].items(), key=lambda item: (-item[1], item[0]))],
        ),
        "",
    ]
    if report.get("pairs_equal_after_conservative_normalization"):
        lines.extend(
            [
                "Pairs that became byte-identical after the conservative Phase 2 normalization "
                "(an upper bound on what normalization alone can collapse, not a match decision):",
                "",
                _mapping_table(
                    report["pairs_equal_after_conservative_normalization"], "field", "pairs"
                ),
                "",
            ]
        )
    for category, examples in sorted(report.get("examples", {}).items()):
        if not examples:
            continue
        lines.append(f"### `{category}`")
        lines.append("")
        for example in examples:
            lines.append(
                f"- `{example['left_id']}`: `{_code(example['left_value'])}`  \n"
                f"  `{example['right_id']}`: `{_code(example['right_value'])}`  \n"
                f"  detail: {example['detail']}"
            )
        lines.append("")
    return lines


def _section_overlap(profile: Mapping[str, Any]) -> list[str]:
    overlap = profile.get("entity_overlap")
    if not overlap:
        return []
    lines = [
        "## 11. Duplicate and Overlapping Entities",
        "",
        _table(
            ["comparison", "shared normalized name+address keys", "share of smaller side"],
            [
                [name, _text(entry["shared"]), _text(entry["share_of_smaller_percent"])]
                for name, entry in sorted(overlap["pairs"].items())
            ],
        ),
        "",
    ]
    for name, entry in sorted(overlap["pairs"].items()):
        lines.append(f"- `{name}`: {_example_list(entry['examples'], 4)}")
    lines.append("")
    return lines


def _section_leakage(profile: Mapping[str, Any]) -> list[str]:
    leakage = profile.get("leakage")
    if not leakage:
        return []
    lines = [
        "## 12. Leakage and Duplication Risks",
        "",
        _table(
            ["risk", "measured value", "assessment"],
            [[item["risk"], _text(item["value"]), item["assessment"]] for item in leakage["risks"]],
        ),
        "",
    ]
    if leakage.get("notes"):
        lines.extend([*leakage["notes"], ""])
    return lines


def _section_validation(profile: Mapping[str, Any]) -> list[str]:
    strategy = profile.get("validation_strategy")
    if not strategy:
        return []
    return [
        "## 13. Recommended Entity-Level Validation Strategy",
        "",
        *strategy["recommendations"],
        "",
    ]


def _section_methodology(profile: Mapping[str, Any]) -> list[str]:
    method = profile["methodology"]
    lines = [
        "## 14. Methodology and Measurement Notes",
        "",
        f"- Distinct-value counting: `{method['distinct_value_method']}` "
        "(exact distinct counting on a 64-bit content hash; the probability of a "
        "collision between two different values is below 1e-5 at this data size).",
        f"- Row counts, null counts, string lengths, country labels and identifier structure: `{method['exact_counters']}`.",
        f"- Exploratory pattern battery: `{method['pattern_battery']}`.",
        f"- Test split: `{method['test_split_policy']}`.",
        f"- Ground truth: `{method['ground_truth_policy']}`.",
        f"- Chunksize: {_text(method['chunksize'])} rows.",
        f"- Raw data modified: `{method['raw_data_modified']}`.",
    ]
    return lines


def render_html(profile: Mapping[str, Any], markdown_text: str | None = None) -> str:
    """Render a self-contained HTML review report from the same measured data."""

    title = "Amazon ML Challenge 2026 - Phase 1 Data Report"
    body_sections: list[str] = []
    for name, items in (
        ("Scope and file discovery", _section_scope(profile)),
        ("Row counts and duplicates", _section_files(profile)),
        ("Schemas", _section_schemas(profile)),
        ("Missingness", _section_missingness(profile)),
        ("Identifier and source structure", _section_identifiers(profile)),
        ("Country distribution", _section_countries(profile)),
        ("Ground truth", _section_ground_truth(profile)),
        ("Match cardinality", _section_cardinality(profile)),
        ("Observed noise patterns", _section_noise(profile)),
        ("Linked-pair variations", _section_variations(profile)),
        ("Entity overlap", _section_overlap(profile)),
        ("Leakage risks", _section_leakage(profile)),
        ("Validation strategy", _section_validation(profile)),
        ("Methodology", _section_methodology(profile)),
    ):
        body_sections.append(f"<h2>{html.escape(name)}</h2>{_markdown_block(items)}")

    generated = html.escape(str(profile["generated_at"]))
    style = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
           margin: 2rem auto; max-width: 1200px; color: #1b1b1b; line-height: 1.5; }
    h1 { border-bottom: 3px solid #232f3e; padding-bottom: .3rem; }
    h2 { margin-top: 2.5rem; color: #232f3e; border-bottom: 1px solid #ddd; }
    table { border-collapse: collapse; margin: .8rem 0; font-size: .9rem; }
    th, td { border: 1px solid #ccc; padding: .35rem .6rem; text-align: left; vertical-align: top; }
    th { background: #f2f4f6; }
    code { background: #f4f4f4; padding: .1rem .3rem; border-radius: 3px; }
    .meta { color: #555; font-size: .9rem; }
    """
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n<style>{style}</style>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f'<p class="meta">Generated {generated} from the real training files. '
        "Every value on this page was measured from the dataset.</p>\n"
        + "\n".join(body_sections)
        + "\n</body>\n</html>\n"
    )


def _markdown_block(items: Sequence[str]) -> str:
    """Convert the small markdown subset used by the report into HTML."""

    out: list[str] = []
    in_table = False
    in_list = False
    flat_lines: list[str] = []
    for item in items:
        flat_lines.extend(str(item).splitlines())
    for raw in flat_lines:
        line = raw.rstrip()
        if not line:
            if in_table:
                out.append("</table>")
                in_table = False
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} and cell for cell in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{_inline(cell)}</th>" for cell in cells) + "</tr>")
                continue
            out.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        heading = line.lstrip("#").strip()
        level = len(line) - len(line.lstrip("#"))
        if heading and set(heading) != {"-"} and 1 <= level <= 4 and line.startswith("#"):
            out.append(f"<h{level}>{_inline(heading)}</h{level}>")
            continue
        out.append(f"<p>{_inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _inline(text: str) -> str:
    """Escape text and convert the markdown inline subset used by the report."""

    escaped = html.escape(text)
    parts = escaped.split("`")
    rendered = "".join(
        part if index % 2 == 0 else f"<code>{part}</code>"
        for index, part in enumerate(parts)
    )
    rendered = rendered.replace("  \n", "<br>")
    return rendered


def write_json(path, payload: Mapping[str, Any]) -> None:
    """Write a JSON artifact with stable key order and UTF-8 content."""

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def render_yaml_rules(derived: Mapping[str, Any]) -> str:
    """Render the derived rule tables as a YAML block for review.

    The output is a proposal for a human to copy into ``config.yaml``; nothing
    writes configuration automatically.
    """

    lines = ["# Derived from the training sources only. Review before use.", "normalization:"]
    name = derived.get("business_name", {})
    address = derived.get("business_address", {})

    lines.append("  name:")
    lines.append("    general_legal_suffixes:")
    for variant, canonical in (name.get("proposed_legal_form_synonyms") or {}).items():
        lines.append(f'      "{variant}": "{canonical}"')
    lines.append("    abbreviations: {}")

    lines.append("  address:")
    for key, source in (
        ("street_designators", "proposed_street_designators"),
        ("unit_markers", "proposed_unit_markers"),
        ("direction_tokens", "proposed_direction_tokens"),
    ):
        lines.append(f"    {key}:")
        table = address.get(source) or {}
        if not table:
            lines[-1] = f"    {key}: {{}}"
            continue
        for variant, canonical in table.items():
            lines.append(f'      "{variant}": "{canonical}"')
    markers = address.get("proposed_missing_markers") or {}
    lines.append("    missing_markers:")
    if not markers:
        lines[-1] = "    missing_markers: []"
    else:
        lines[-1] = (
            "    missing_markers: ["
            + ", ".join(f'"{token}"' for token in sorted(markers))
            + "]"
        )
    lines.append("    address_tokens: {}")
    return "\n".join(lines) + "\n"
