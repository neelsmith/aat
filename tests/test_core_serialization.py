"""Offline tests for aat.core.serialization -- no dspy or network needed."""

import pytest

from aat.core import AATNode
from aat.core.serialization import read_graph, read_nodes, serialize_nodes, write_nodes

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
