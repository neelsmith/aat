"""Offline tests for aat.english.html.tokens_to_html() -- no dspy needed."""

from aat.core import AATGraph, AATNode, CitableToken
from aat.core.coloring import assign_action_colors
from aat.english.html import tokens_to_html


def _tok(context, id, value):
    return CitableToken(context=context, id=id, value=value)


def test_ordinary_words_are_space_separated():
    tokens = [_tok("c1", "t1", "The"), _tok("c1", "t2", "dog"), _tok("c1", "t3", "ran")]
    assert tokens_to_html(tokens) == "The dog ran"


def test_punctuation_attaches_to_preceding_word():
    tokens = [
        _tok("c1", "t1", "Hello"),
        _tok("c1", "t2", ","),
        _tok("c1", "t3", "world"),
        _tok("c1", "t4", "!"),
    ]
    assert tokens_to_html(tokens) == "Hello, world!"


def test_opening_bracket_right_joins_and_closing_bracket_left_joins():
    tokens = [
        _tok("c1", "t1", "He"),
        _tok("c1", "t2", "said"),
        _tok("c1", "t3", "("),
        _tok("c1", "t4", "quietly"),
        _tok("c1", "t5", ")"),
        _tok("c1", "t6", "."),
    ]
    assert tokens_to_html(tokens) == "He said (quietly)."


def test_quote_pair_alternates_right_then_left_joining():
    tokens = [
        _tok("c1", "t1", "She"),
        _tok("c1", "t2", "said"),
        _tok("c1", "t3", '"'),
        _tok("c1", "t4", "Hi"),
        _tok("c1", "t5", '"'),
        _tok("c1", "t6", "."),
    ]
    # The quote character itself is HTML-escaped (html.escape's default
    # quote=True) same as any other character -- the join *positions*
    # (opening quote right-joins, closing quote left-joins, per this
    # module's docstring) are what this test is really checking.
    assert tokens_to_html(tokens) == 'She said &quot;Hi&quot;.'


def test_second_quote_pair_in_same_passage_alternates_again():
    tokens = [
        _tok("c1", "t1", '"'),
        _tok("c1", "t2", "Hi"),
        _tok("c1", "t3", '"'),
        _tok("c1", "t4", "and"),
        _tok("c1", "t5", '"'),
        _tok("c1", "t6", "Bye"),
        _tok("c1", "t7", '"'),
    ]
    assert tokens_to_html(tokens) == '&quot;Hi&quot; and &quot;Bye&quot;'


def test_special_characters_are_html_escaped():
    tokens = [_tok("c1", "t1", "Tom"), _tok("c1", "t2", "&"), _tok("c1", "t3", "Jerry")]
    rendered = tokens_to_html(tokens)
    # "&" isn't a bracket or quote char, so it defaults to left-joining
    # (same as a comma or period would) -- no space before it, per this
    # module's own docstring on punctuation's default join behavior.
    assert rendered == "Tom&amp; Jerry"


def test_lone_angle_bracket_token_is_escaped_not_left_as_markup():
    tokens = [_tok("c1", "t1", "a"), _tok("c1", "t2", "<"), _tok("c1", "t3", "b")]
    rendered = tokens_to_html(tokens)
    assert "&lt;" in rendered
    assert "<" not in rendered.replace("&lt;", "")


def test_graph_none_gives_plain_unhighlighted_text():
    tokens = [_tok("c1", "t1", "The"), _tok("c1", "t2", "dog"), _tok("c1", "t3", "ran")]
    assert tokens_to_html(tokens, graph=None) == "The dog ran"
    assert "<span" not in tokens_to_html(tokens, graph=None)


def _dog_ate_homework():
    tokens = [
        _tok("c1", "t1", "The"),
        _tok("c1", "t2", "dog"),
        _tok("c1", "t3", "ate"),
        _tok("c1", "t4", "the"),
        _tok("c1", "t5", "homework"),
        _tok("c1", "t6", "."),
    ]
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
            AATNode(context="c1", id="t5", value="homework", role="target", related_node="t3"),
        ]
    )
    return tokens, graph


def test_tokens_not_in_the_graph_are_left_unhighlighted():
    tokens, graph = _dog_ate_homework()
    rendered = tokens_to_html(tokens, graph=graph)
    # "The", "the", and "." have no corresponding node -- plain text, no span.
    assert ">The<" not in rendered
    assert rendered.startswith("The ")
    assert rendered.endswith(".")


def test_action_agent_and_target_get_the_role_specific_borders():
    tokens, graph = _dog_ate_homework()
    rendered = tokens_to_html(tokens, graph=graph)

    # action ("ate"): boxed, sharp corners.
    assert "border: 2px solid" in rendered
    assert "border-radius: 2px" in rendered
    # agent ("dog"): rounded box.
    assert "border-radius: 0.8em" in rendered
    # target ("homework"): underline only.
    assert "border-bottom: 3px solid" in rendered


def test_highlighted_tokens_share_the_mermaid_diagram_colors():
    tokens, graph = _dog_ate_homework()
    colors, _warnings = assign_action_colors(graph)
    fill, stroke, text_color = colors[("c1", "t3")]

    rendered = tokens_to_html(tokens, graph=graph)
    assert f"background-color: {fill}" in rendered
    assert f"color: {text_color}" in rendered
    assert stroke in rendered


def test_dependent_action_gets_its_own_cluster_color_in_html_too():
    tokens = [
        _tok("c1", "t1", "He"),
        _tok("c1", "t2", "said"),
        _tok("c1", "t3", "the"),
        _tok("c1", "t4", "dog"),
        _tok("c1", "t5", "ate"),
        _tok("c1", "t6", "."),
    ]
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t2", value="said", role="action", related_node=None),
            AATNode(context="c1", id="t1", value="He", role="agent", related_node="t2"),
            AATNode(context="c1", id="t5", value="ate", role="action", related_node="t2"),
            AATNode(context="c1", id="t4", value="dog", role="agent", related_node="t5"),
        ]
    )
    colors, _warnings = assign_action_colors(graph)
    rendered = tokens_to_html(tokens, graph=graph)
    said_fill = colors[("c1", "t2")][0]
    ate_fill = colors[("c1", "t5")][0]
    assert said_fill != ate_fill
    assert f"background-color: {said_fill}" in rendered
    assert f"background-color: {ate_fill}" in rendered
