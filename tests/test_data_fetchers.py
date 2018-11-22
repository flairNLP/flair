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


def test_load_no_dev_data(tasks_base_path):
    # get training, test and dev data
    corpus = NLPTaskDataFetcher.fetch_column_corpus(tasks_base_path / 'fashion_nodev', {0: 'text', 2: 'ner'})

    assert len(corpus.train) == 5
    assert len(corpus.dev) == 1
    assert len(corpus.test) == 1


def test_multi_corpus(tasks_base_path):
    # get two corpora as one
    corpus = NLPTaskDataFetcher.fetch_corpora([NLPTask.FASHION, NLPTask.GERMEVAL])

    assert len(corpus.train) == 8
    assert len(corpus.dev) == 2
    assert len(corpus.test) == 2