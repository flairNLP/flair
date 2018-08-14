from flair.data_fetcher import NLPTask, NLPTaskDataFetcher


def test_load_imdb_data():
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)

    assert len(corpus.train) == 5
    assert len(corpus.dev) == 5
    assert len(corpus.test) == 5


def test_load_ag_news_data():
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.AG_NEWS)

    assert len(corpus.train) == 10
    assert len(corpus.dev) == 10
    assert len(corpus.test) == 10


def test_load_sequence_labeling_data():
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION)

    assert len(corpus.train) == 6
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_load_germeval_data():
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.GERMEVAL)

    assert len(corpus.train) == 2
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_load_ud_english_data():
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.UD_ENGLISH)

    assert len(corpus.train) == 6
    assert len(corpus.test) == 4
    assert len(corpus.dev) == 2