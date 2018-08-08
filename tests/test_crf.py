from flair.models import SequenceTagger, SequenceTaggerOld
from flair.embeddings import WordEmbeddings
from flair.data import Sentence, Dictionary
import torch
import unittest
import time


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])

    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))

    return maxi + recti_


dict_ = Dictionary()

dict_.add_item('Loc')
dict_.add_item('None')

print('getting glove embeddings...')

emb = WordEmbeddings('glove')

st = SequenceTagger(32, emb, dict_, 'ner')
sto = SequenceTaggerOld(32, emb, dict_, 'ner')

st.linear = sto.linear
st.transitions = sto.transitions
st.embedding2nn = sto.embedding2nn
st.rnn = sto.rnn

sto.eval()
st.eval()

sentence = Sentence('I love Berlin .')


class Test1(unittest.TestCase):
    def test(self):

        features = torch.randn(5, 3)

        alpha_old = sto._forward_alg(features)
        alpha = st._forward_alg([features, features])

        print(alpha)
        print(alpha_old)


class Test2(unittest.TestCase):
    def test(self):

        x = torch.randn(2, 3)

        out11 = log_sum_exp(x[0, :][None, :])
        out12 = log_sum_exp(x[1, :][None, :])

        out2 = log_sum_exp_batch(x)

        self.assertEqual(out11, out2[0])
        self.assertEqual(out12, out2[1])


class Test3(unittest.TestCase):
    def test(self):

        feats = torch.randn(5, 3)
        tags = torch.LongTensor(5).random_(0, 3)

        out_old = sto._score_sentence(feats, tags)

        out = st._score_sentence(
            torch.stack([feats, feats]),
            torch.stack([tags, tags]),
        )

        rel_diff = abs((out[0].item() - out_old.item()) / out[0].item())

        self.assertLess(rel_diff, 0.001)


class Test4(unittest.TestCase):
    def test(self):

        out_old = sto.neg_log_likelihood([sentence, sentence], 'ner')
        out = st.neg_log_likelihood([sentence, sentence], 'ner')

        rel_diff = abs((out_old.item() - out.item()) / out.item())

        self.assertLess(rel_diff, 0.001)


class Test5(unittest.TestCase):
    def test(self):

        feats = torch.randn(1000, 5, 3)

        start_ = time.time()

        for i in range(feats.shape[0]):
            sto._forward_alg(feats[i])

        stop_old = time.time() - start_

        print('Time elapsed: {}s'.format(stop_old))

        start_ = time.time()
        st._forward_alg(feats)
        stop_ = time.time() - start_

        print('Time elapsed: {}s'.format(stop_))

        print('Speedup factor is {}'.format(stop_old / stop_))


if __name__ == '__main__':
    unittest.main()