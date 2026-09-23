"""Offline tests for aat.core.serialization -- no dspy or network needed."""

import pytest

from aat.core import AATNode, CitableToken
from aat.core.serialization import (
    read_analysis,
    read_graph,
    read_nodes,
    read_tokens,
    serialize_analysis,
    serialize_nodes,
    serialize_tokens,
    write_analysis,
    write_nodes,
    write_tokens,
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


_TOKENS = [
    CitableToken(context="c1", id="t1", value="The"),
    CitableToken(context="c1", id="t2", value="dog"),
    CitableToken(context="c1", id="t3", value="ate"),
    CitableToken(context="c1", id="t4", value="the"),
    CitableToken(context="c1", id="t5", value="homework"),
    CitableToken(context="c1", id="t6", value="."),
]


def test_serialize_tokens_shape():
    text = serialize_tokens(_TOKENS)
    lines = text.splitlines()
    assert lines[0] == "#!tokens"
    assert lines[1] == "context|id|value"
    assert lines[2] == "c1|t1|The"
    assert lines[3] == "c1|t2|dog"
    assert lines[-1] == "c1|t6|."


def test_serialize_tokens_preserves_order():
    # Row order is the only thing that records reading order for a
    # '#!tokens' block -- confirm it isn't silently re-sorted by id or
    # anything else.
    text = serialize_tokens(list(reversed(_TOKENS)))
    ids_in_file_order = [line.split("|")[1] for line in text.splitlines()[2:]]
    assert ids_in_file_order == [t.id for t in reversed(_TOKENS)]


def test_write_and_read_tokens_roundtrip(tmp_path):
    path = tmp_path / "tokens.txt"
    write_tokens(_TOKENS, str(path))
    roundtripped = read_tokens(str(path))
    assert roundtripped == _TOKENS


def test_read_tokens_rejects_file_with_no_block(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("not a valid tokens file\n")
    with pytest.raises(ValueError):
        read_tokens(str(path))


def test_read_nodes_ignores_a_tokens_block_in_the_same_file(tmp_path):
    path = tmp_path / "combo.txt"
    path.write_text(serialize_tokens(_TOKENS) + "\n" + serialize_nodes(_NODES))
    roundtripped = read_nodes(str(path))
    assert roundtripped == _NODES


def test_read_tokens_ignores_an_aatnodes_block_in_the_same_file(tmp_path):
    path = tmp_path / "combo.txt"
    path.write_text(serialize_tokens(_TOKENS) + "\n" + serialize_nodes(_NODES))
    roundtripped = read_tokens(str(path))
    assert roundtripped == _TOKENS


def test_read_nodes_still_requires_its_own_block_even_if_tokens_present(tmp_path):
    path = tmp_path / "tokens_only.txt"
    write_tokens(_TOKENS, str(path))
    with pytest.raises(ValueError):
        read_nodes(str(path))


def test_write_and_read_analysis_roundtrip(tmp_path):
    from aat.core import AATGraph

    path = tmp_path / "analysis.txt"
    write_analysis(_TOKENS, AATGraph(nodes=_NODES), str(path))
    tokens, graph = read_analysis(str(path))
    assert tokens == _TOKENS
    assert [n.id for n in graph.nodes] == ["t3", "t2", "t5"]


def test_serialize_analysis_is_tokens_block_then_nodes_block():
    from aat.core import AATGraph

    text = serialize_analysis(_TOKENS, AATGraph(nodes=_NODES))
    tokens_part = serialize_tokens(_TOKENS)
    nodes_part = serialize_nodes(_NODES)
    assert text == tokens_part + "\n" + nodes_part


def test_write_analysis_is_a_thin_wrapper_around_serialize_analysis(tmp_path):
    from aat.core import AATGraph

    graph = AATGraph(nodes=_NODES)
    path = tmp_path / "analysis.txt"
    write_analysis(_TOKENS, graph, str(path))
    assert path.read_text() == serialize_analysis(_TOKENS, graph)


def test_serialize_analysis_returns_a_string_with_no_file_written(tmp_path):
    from aat.core import AATGraph

    # serialize_analysis() takes no path at all -- calling it can't have
    # written anything to disk, unlike write_analysis().
    text = serialize_analysis(_TOKENS, AATGraph(nodes=_NODES))
    assert isinstance(text, str)
    assert list(tmp_path.iterdir()) == []


def test_read_analysis_round_trips_a_sentence_spanning_composite_id(tmp_path):
    # The whole point of this format change: a composite id like
    # "1.14.t3" (aat.english.sentences' own scheme for a sentence
    # spanning several citation units) is just data here, not something
    # that needs re-deriving by re-running tokenize_corpus_by_sentence()
    # -- confirm it round-trips completely unremarkably.
    from aat.core import AATGraph

    tokens = [
        CitableToken(context="urn:cts:test:work:1.14-1.15", id="1.14.t1", value="and"),
        CitableToken(context="urn:cts:test:work:1.14-1.15", id="1.15.t1", value="ruled"),
    ]
    nodes = [
        AATNode(
            context="urn:cts:test:work:1.14-1.15",
            id="1.15.t1",
            value="ruled",
            role="action",
            related_node=None,
        ),
    ]
    path = tmp_path / "spanning.txt"
    write_analysis(tokens, AATGraph(nodes=nodes), str(path))

    reloaded_tokens, reloaded_graph = read_analysis(str(path))
    assert reloaded_tokens == tokens
    assert [n.id for n in reloaded_graph.nodes] == ["1.15.t1"]
