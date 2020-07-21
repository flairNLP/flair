import pickle

from flair.data import Sentence, Image, DataTuple, FlairDataset


class ItemsDataset(FlairDataset):

    def __init__(self,
                 items_filename,
                 item_parts
                 ):
        self.data = pickle.load(open(items_filename, 'rb'))
        self.item_parts = item_parts

    def __getitem__(self, index):
        if len(self.item_parts) > 1:
            return DataTuple([part_data(self.data[index][part_name])
                              for part_name, part_data in
                              self.item_parts])
        else:
            part_name, part_data = self.item_parts[0]
            return part_data(self.data[index][part_name])

    def __len__(self):
        return len(self.data)


class QueriesDataset(ItemsDataset):

    def __init__(self,
                 queries_filename,
                 queries_parts=[('query', Sentence)]
                 ):
        super().__init__(queries_filename, queries_parts)

    def __str__(self):
        return f'n_queries={len(self)}'


class ArticlesDataset(ItemsDataset):

    def __init__(self,
                 articles_filename,
                 article_parts=[('metadata', Sentence),
                                ('image', Image)]
                 ):
        super().__init__(articles_filename, article_parts)
        self.sku_map = {self.data[index]['sku']: index
                        for index in range(len(self.data))}

    def id2sku(self, index):
        return self.data[index]['sku']

    def sku2id(self, sku):
        return self.sku_map[sku]

    def __str__(self):
        return f'n_articles={len(self)}'
