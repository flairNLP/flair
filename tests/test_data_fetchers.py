from flair.data_fetcher import NLPTask, NLPTaskDataFetcher


def test_load_imdb_data(tasks_base_path):
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.IMDB, tasks_base_path)

    assert len(corpus.train) == 5
    assert len(corpus.dev) == 5
    assert len(corpus.test) == 5


def test_load_ag_news_data(tasks_base_path):
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.AG_NEWS, tasks_base_path)

    assert len(corpus.train) == 10
    assert len(corpus.dev) == 10
    assert len(corpus.test) == 10


def test_load_sequence_labeling_data(tasks_base_path):
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.FASHION, tasks_base_path)

    assert len(corpus.train) == 6
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_load_germeval_data(tasks_base_path):
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.GERMEVAL, tasks_base_path)

    assert len(corpus.train) == 2
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_load_ud_english_data(tasks_base_path):
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_data(NLPTask.UD_ENGLISH, tasks_base_path)

    assert len(corpus.train) == 6
    assert len(corpus.test) == 4
    assert len(corpus.dev) == 2
