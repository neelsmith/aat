"""Offline tests for aat_main.py's serialization helper -- no dspy LM
call and no .env needed (aat_main only reads env vars inside
_configure_lm(), which this test never calls). See TESTING.md."""

from aat.core import AATGraph, AATNode, CitableToken, read_analysis, serialize_analysis

import aat_main


def _dog_ate_homework_tokens():
    return [
        CitableToken(context="ex.1", id="t1", value="The"),
        CitableToken(context="ex.1", id="t2", value="dog"),
        CitableToken(context="ex.1", id="t3", value="ate"),
        CitableToken(context="ex.1", id="t4", value="my"),
        CitableToken(context="ex.1", id="t5", value="homework"),
        CitableToken(context="ex.1", id="t6", value="."),
    ]


def _dog_ate_homework_graph():
    return AATGraph(
        nodes=[
            AATNode(context="ex.1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="ex.1", id="t2", value="dog", role="agent", related_node="t3"),
            AATNode(context="ex.1", id="t5", value="homework", role="target", related_node="t3"),
        ]
    )


def test_write_serialized_analysis_matches_serialize_analysis_exactly(capsys):
    tokens = _dog_ate_homework_tokens()
    graph = _dog_ate_homework_graph()

    aat_main._write_serialized_analysis(tokens, graph)

    captured = capsys.readouterr()
    assert captured.out == serialize_analysis(tokens, graph)
    assert captured.err == ""


def test_stdout_is_exactly_the_serialized_blocks_nothing_else(capsys):
    # Guards against a future change reintroducing a human-readable print
    # (like the old _print_graph()) alongside the serialized output --
    # stdout has to be *only* the '#!tokens'/'#!aatnodes' blocks so it
    # can be redirected straight to a file and reloaded.
    tokens = _dog_ate_homework_tokens()
    graph = _dog_ate_homework_graph()

    aat_main._write_serialized_analysis(tokens, graph)

    captured = capsys.readouterr()
    assert captured.out.startswith("#!tokens\n")
    assert captured.out.count("#!tokens") == 1
    assert captured.out.count("#!aatnodes") == 1


def test_stdout_output_round_trips_through_read_analysis(capsys, tmp_path):
    tokens = _dog_ate_homework_tokens()
    graph = _dog_ate_homework_graph()

    aat_main._write_serialized_analysis(tokens, graph)
    captured = capsys.readouterr()

    saved = tmp_path / "analysis.txt"
    saved.write_text(captured.out, encoding="utf-8")

    reloaded_tokens, reloaded_graph = read_analysis(str(saved))
    assert reloaded_tokens == tokens
    assert reloaded_graph == graph


def test_empty_context_round_trips_too():
    # args.context defaults to "" (see argparse setup in aat_main.py) --
    # confirm an empty-string context serializes and reloads correctly,
    # not just a real citable reference.
    tokens = [CitableToken(context="", id="t1", value="Hi"), CitableToken(context="", id="t2", value=".")]
    graph = AATGraph(nodes=[])
    import io
    import sys

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        aat_main._write_serialized_analysis(tokens, graph)
    finally:
        sys.stdout = old_stdout

    assert buf.getvalue() == serialize_analysis(tokens, graph)
