from flair.data_fetcher import NLPTask, NLPTaskDataFetcher


def test_loading_imdb_data():
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB)

    assert len(corpus.train) == 5
    assert len(corpus.dev) == 5
    assert len(corpus.test) == 5


def test_loading_ag_news_data():
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.AG_NEWS)

    assert len(corpus.train) == 10
    assert len(corpus.dev) == 10
    assert len(corpus.test) == 10