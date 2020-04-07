from datasets.biomedical import (
    Entity,
    find_overlapping_entities,
    find_nested_entities,
    normalize_entity_spans,
)


def test_find_overlapping_entities():
    not_overlapping_entities = [Entity((0, 3), ""), Entity((5, 10), "")]
    result = find_overlapping_entities(not_overlapping_entities)
    assert len(result) == 0

    not_overlapping_entities = [Entity((0, 3), ""), Entity((3, 6), "")]
    result = find_overlapping_entities(not_overlapping_entities)
    assert len(result) == 0

    entity1 = Entity((0, 6), "")
    entity2 = Entity((3, 7), "")
    entity3 = Entity((10, 12), "")
    simple_overlapping_entities = [entity1, entity2, entity3]
    result = find_overlapping_entities(simple_overlapping_entities)
    assert len(result) == 1
    assert result[0] == (entity1, entity2)

    entity1 = Entity((0, 6), "")
    entity2 = Entity((3, 6), "")
    entity3 = Entity((4, 8), "")
    entity4 = Entity((7, 10), "")
    entity5 = Entity((12, 13), "")
    simple_overlapping_entities = [entity1, entity2, entity3, entity4, entity5]
    result = find_overlapping_entities(simple_overlapping_entities)
    assert len(result) == 4
    assert result[0] == (entity1, entity2)
    assert result[1] == (entity1, entity3)
    assert result[2] == (entity2, entity3)
    assert result[3] == (entity3, entity4)


def test_find_nested_entities():
    entity1 = Entity((0, 12), "a")
    entity2 = Entity((1, 4), "a")
    entity3 = Entity((5, 6), "a")

    result = find_nested_entities([entity3, entity1, entity2])
    assert len(result) == 1
    assert result[0].char_span == range(0, 12)
    assert result[0].type == "a"
    assert len(result[0].nested_entities) == 2
    assert entity2 in result[0].nested_entities
    assert entity3 in result[0].nested_entities


def test_normalize_entity_spans():
    entity1 = Entity((0, 12), "")
    entity2 = Entity((1, 4), "")
    entity3 = Entity((5, 6), "")

    result = normalize_entity_spans([entity2, entity1, entity3])
    assert len(result) == 2
    assert result[0] == entity2
    assert result[1] == entity3

    entity1 = Entity((0, 20), "")
    entity2 = Entity((0, 10), "")
    entity3 = Entity((10, 20), "")
    entity4 = Entity((2, 8), "")
    entity5 = Entity((12, 18), "")

    result = normalize_entity_spans([entity2, entity5, entity1, entity3, entity4])
    assert len(result) == 2
    assert result[0] == entity2
    assert result[1] == entity3

    entity1 = Entity((0, 12), "")
    entity2 = Entity((1, 4), "")
    entity3 = Entity((5, 6), "")
    entity4 = Entity((8, 16), "")

    result = normalize_entity_spans([entity2, entity1, entity4, entity3])
    assert len(result) == 3
    assert result[0] == entity2
    assert result[1] == entity3
    assert result[2].char_span == range(12, 16)
