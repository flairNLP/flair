import pytest
from flair.models import TARSSequenceTagger2
from flair.data import Dictionary, Sentence

@pytest.fixture
def tagger():
    tag_type = "ner"
    tag_dictionary = Dictionary(add_unk=False)
    tag_dictionary.add_item("O")
    tag_dictionary.add_item("B-Test_Case")
    tag_dictionary.add_item("I-Test_Case")
    task_name = "TEST_TASK_NAME"
    return TARSSequenceTagger2(tag_dictionary=tag_dictionary, tag_type=tag_type, task_name=task_name)

def test_get_cleaned_up_tag(tagger):
    cleaned_up_tag = tagger._get_cleaned_up_tag("B-Test_Case")
    assert cleaned_up_tag == "B-Test Case"

def test_split_tag_O(tagger):
    prefix, tag = tagger._split_tag("O")
    assert prefix == "O"
    assert tag is None

def test_split_tag_B_TestCase(tagger):
    prefix, tag = tagger._split_tag("B-Test Case")
    assert prefix == "B"
    assert tag == "Test Case"

def test_split_tag_empty_string(tagger):
    prefix, tag = tagger._split_tag("")
    assert prefix is None
    assert tag is None

def test_get_tag_dictionary_no_prefix(tagger):
    tag_dictionary_no_prefix = tagger._get_tag_dictionary_no_prefix()
    assert len(tag_dictionary_no_prefix.idx2item) == 1
    assert tag_dictionary_no_prefix.idx2item[0].decode("utf-8") == "Test_Case"

def test_compute_tag_similarity_for_current_epoch(tagger):
    tagger._compute_tag_similarity_for_current_epoch()
    assert len(tagger.tag_nearest_map) == len(tagger.tag_dictionary_no_prefix.idx2item)

def test_get_random_tags(tagger):
    random_set = tagger._get_random_tags()
    assert len(random_set) == 1  # only Test_Case can be sampled
    assert next(iter(random_set)) == "Test_Case"

def test_get_tars_formatted_sentences(tagger):
    sentences = [Sentence("This is a fancy test.")]
    sentences[0][2].add_tag(tagger.tag_type, "O")
    sentences[0][3].add_tag(tagger.tag_type, "B-Test_Case")
    sentences[0][4].add_tag(tagger.tag_type, "I-Test_Case")
    tagger.train()  # triggers _compute_tag_similarity_for_current_epoch
    formatted_sentences = tagger._get_tars_formatted_sentences(sentences)
    assert len(formatted_sentences) == 1
    assert formatted_sentences[0][0].text == "Test"
    assert formatted_sentences[0][1].text == "Case"
    assert formatted_sentences[0][2].text == "[SEP]"
    assert formatted_sentences[0][3].text == "This"
    assert formatted_sentences[0][4].text == "is"
    assert formatted_sentences[0][5].text == "a"
    assert formatted_sentences[0][6].text == "fancy"
    assert formatted_sentences[0][7].text == "test"
    assert formatted_sentences[0][8].text == "."

    assert formatted_sentences[0][5].get_tag(tagger.static_tag_type).value == tagger.static_tag_outside
    assert formatted_sentences[0][6].get_tag(tagger.static_tag_type).value == tagger.static_tag_beginning
    assert formatted_sentences[0][7].get_tag(tagger.static_tag_type).value == tagger.static_tag_inside

def test_make_ad_hoc_tag_dictionary(tagger):
    ad_hoc_dict = tagger._make_ad_hoc_tag_dictionary(["O", "B-TEST_DICT", "I-TEST_DICT"])
    assert len(ad_hoc_dict.idx2item) == 3
