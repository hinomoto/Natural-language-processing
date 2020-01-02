import pickle
from gensim.models import word2vec

with open('txtresult.pickle', mode='rb') as f:
    wagahai_words = pickle.load(f)

print(wagahai_words)

# size : 中間層のニューロン数
# min_count : この値以下の出現回数の単語を無視
# window : 対象単語を中心とした前後の単語数
# iter : epochs数
# sg : CBOWを使うかskip-gramを使うか 0:CBOW 1:skip-gram
model = word2vec.Word2Vec(wagahai_words,
                          size=100,
                          min_count=5,
                          window=5,
                          iter=20,
                          sg = 0)

a=model.wv.most_similar(positive=["量", "価"])
print(a)

b=model.wv.most_similar(positive=["量", "価"], negative=["世"])
print(b)
#negativeは引く単語のリストです.