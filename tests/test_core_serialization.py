"""Offline tests for aat.core.serialization -- no dspy or network needed."""

import pytest

from aat.core import AATNode, CitedPassage
from aat.core.serialization import (
    read_analysis,
    read_graph,
    read_nodes,
    read_passages,
    serialize_analysis,
    serialize_nodes,
    serialize_passages,
    write_analysis,
    write_nodes,
    write_passages,
)

_NODES = [
    AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
    AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
    AATNode(context="c1", id="t5", value="homework", role="target", related_node="t3"),
]


def test_serialize_nodes_shape():
    text = serialize_nodes(_NODES)
    lines = text.splitlines()
    assert lines[0] == "#!aatnodes"
    assert lines[1] == "context|id|value|role|related_node"
    assert lines[2] == "c1|t3|ate|action|"
    assert lines[3] == "c1|t2|dog|agent|t3"


def test_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "graph.txt"
    write_nodes(_NODES, str(path))
    roundtripped = read_nodes(str(path))
    assert roundtripped == _NODES


def test_read_graph_wraps_as_aatgraph(tmp_path):
    path = tmp_path / "graph.txt"
    write_nodes(_NODES, str(path))
    graph = read_graph(str(path))
    assert [n.id for n in graph.nodes] == ["t3", "t2", "t5"]


def test_read_nodes_rejects_file_with_no_block(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("not a valid aatnodes file\n")
    with pytest.raises(ValueError):
        read_nodes(str(path))


def test_read_nodes_rejects_bad_header(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("#!aatnodes\nwrong|header\n")
    with pytest.raises(ValueError):
        read_nodes(str(path))


def test_read_nodes_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("#!aatnodes\ncontext|id|value|role|related_node\nc1|t1|only-four-cols|action\n")
    with pytest.raises(ValueError):
        read_nodes(str(path))


def test_multiple_blocks_concatenate_in_file_order(tmp_path):
    path = tmp_path / "combo.txt"
    text = serialize_nodes(_NODES[:1]) + serialize_nodes(_NODES[1:])
    path.write_text(text)
    roundtripped = read_nodes(str(path))
    assert [n.id for n in roundtripped] == ["t3", "t2", "t5"]


_PASSAGES = [
    CitedPassage(context="c1", text="The dog ate the homework."),
]


def test_serialize_passages_shape():
    text = serialize_passages(_PASSAGES)
    lines = text.splitlines()
    assert lines[0] == "#!passages"
    assert lines[1] == "context|text"
    assert lines[2] == "c1|The dog ate the homework."


def test_write_and_read_passages_roundtrip(tmp_path):
    path = tmp_path / "passages.txt"
    write_passages(_PASSAGES, str(path))
    roundtripped = read_passages(str(path))
    assert roundtripped == _PASSAGES


def test_read_passages_rejects_file_with_no_block(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("not a valid passages file\n")
    with pytest.raises(ValueError):
        read_passages(str(path))


def test_read_nodes_ignores_a_passages_block_in_the_same_file(tmp_path):
    path = tmp_path / "combo.txt"
    path.write_text(serialize_passages(_PASSAGES) + "\n" + serialize_nodes(_NODES))
    roundtripped = read_nodes(str(path))
    assert roundtripped == _NODES


def test_read_passages_ignores_an_aatnodes_block_in_the_same_file(tmp_path):
    path = tmp_path / "combo.txt"
    path.write_text(serialize_passages(_PASSAGES) + "\n" + serialize_nodes(_NODES))
    roundtripped = read_passages(str(path))
    assert roundtripped == _PASSAGES


def test_read_nodes_still_requires_its_own_block_even_if_passages_present(tmp_path):
    path = tmp_path / "passages_only.txt"
    write_passages(_PASSAGES, str(path))
    with pytest.raises(ValueError):
        read_nodes(str(path))


def test_write_and_read_analysis_roundtrip(tmp_path):
    from aat.core import AATGraph

    path = tmp_path / "analysis.txt"
    write_analysis(_PASSAGES, AATGraph(nodes=_NODES), str(path))
    passages, graph = read_analysis(str(path))
    assert passages == _PASSAGES
    assert [n.id for n in graph.nodes] == ["t3", "t2", "t5"]


def test_serialize_analysis_is_passages_block_then_nodes_block():
    from aat.core import AATGraph

    text = serialize_analysis(_PASSAGES, AATGraph(nodes=_NODES))
    passages_part = serialize_passages(_PASSAGES)
    nodes_part = serialize_nodes(_NODES)
    assert text == passages_part + "\n" + nodes_part


def test_write_analysis_is_a_thin_wrapper_around_serialize_analysis(tmp_path):
    from aat.core import AATGraph

    graph = AATGraph(nodes=_NODES)
    path = tmp_path / "analysis.txt"
    write_analysis(_PASSAGES, graph, str(path))
    assert path.read_text() == serialize_analysis(_PASSAGES, graph)


def test_serialize_analysis_returns_a_string_with_no_file_written(tmp_path):
    from aat.core import AATGraph

    # serialize_analysis() takes no path at all -- calling it can't have
    # written anything to disk, unlike write_analysis().
    text = serialize_analysis(_PASSAGES, AATGraph(nodes=_NODES))
    assert isinstance(text, str)
    assert list(tmp_path.iterdir()) == []
