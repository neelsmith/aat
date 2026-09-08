"""Offline, subprocess-driven tests for aat_to_dot.py -- no dspy or
network needed (this script only reads aat.core's plain-text format from
stdin and renders it with aat.core.graph_to_dot()). Run as a real
subprocess, not imported, so argparse and the `if __name__ ==
"__main__":` block are actually exercised, not just the helper function.
See TESTING.md."""

import subprocess
import sys
from pathlib import Path

from aat.core import AATGraph, AATNode, CitedPassage, serialize_analysis, serialize_nodes

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "aat_to_dot.py"


def _dog_ate_homework_graph():
    return AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
            AATNode(context="c1", id="t5", value="homework", role="target", related_node="t3"),
        ]
    )


def _run(stdin_text, *args):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )


def test_reads_bare_aatnodes_block_and_writes_dot():
    result = _run(serialize_nodes(_dog_ate_homework_graph().nodes))
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("digraph aat {\n    rankdir=BT;\n")
    assert '    t3 [' in result.stdout
    assert result.stdout.rstrip("\n").endswith("}")


def test_ignores_a_passages_block_alongside_aatnodes():
    # aat_main.py's own stdout (a full serialize_analysis() -- both
    # blocks) is the intended common input; confirm the passages block
    # is accepted and ignored rather than tripping up the parser.
    passage = CitedPassage(context="c1", text="The dog ate my homework.")
    full_analysis = serialize_analysis([passage], _dog_ate_homework_graph())
    result = _run(full_analysis)
    assert result.returncode == 0, result.stderr
    assert "digraph aat" in result.stdout
    assert "#!passages" not in result.stdout


def test_orientation_flag_is_passed_through():
    result = _run(serialize_nodes(_dog_ate_homework_graph().nodes), "--orientation", "LR")
    assert result.returncode == 0, result.stderr
    assert "    rankdir=LR;" in result.stdout


def test_td_orientation_is_mapped_to_graphviz_own_tb_spelling():
    result = _run(serialize_nodes(_dog_ate_homework_graph().nodes), "--orientation", "TD")
    assert result.returncode == 0, result.stderr
    assert "    rankdir=TB;" in result.stdout


def test_no_color_flag_gives_plain_digraph():
    result = _run(serialize_nodes(_dog_ate_homework_graph().nodes), "--no-color")
    assert result.returncode == 0, result.stderr
    assert "fillcolor" not in result.stdout
    assert "digraph aat" in result.stdout


def test_default_is_colored():
    result = _run(serialize_nodes(_dog_ate_homework_graph().nodes))
    assert result.returncode == 0, result.stderr
    assert "fillcolor" in result.stdout


def test_invalid_orientation_exits_nonzero_with_message_on_stderr():
    result = _run(serialize_nodes(_dog_ate_homework_graph().nodes), "--orientation", "sideways")
    assert result.returncode != 0
    assert "sideways" in result.stderr
    assert result.stdout == ""


def test_empty_stdin_exits_nonzero_with_clear_message():
    result = _run("")
    assert result.returncode != 0
    assert "aatnodes" in result.stderr
    assert result.stdout == ""


def test_broken_related_node_warns_on_stderr_not_stdout():
    graph = AATGraph(
        nodes=[AATNode(context="c1", id="t5", value="homework", role="target", related_node="ghost")]
    )
    result = _run(serialize_nodes(graph.nodes))
    assert result.returncode == 0, result.stderr
    assert "ghost" in result.stderr
    assert "ghost" not in result.stdout


def test_leaves_no_temp_file_behind(tmp_path):
    # _read_graph_from_stdin() bridges stdin to aat.core.read_graph() via
    # a throwaway temp file -- confirm it's actually cleaned up, not
    # just "probably fine because os.remove() is in a finally block".
    import tempfile

    before = set(Path(tempfile.gettempdir()).glob("tmp*"))
    result = _run(serialize_nodes(_dog_ate_homework_graph().nodes))
    assert result.returncode == 0, result.stderr
    after = set(Path(tempfile.gettempdir()).glob("tmp*"))
    assert after - before == set()
