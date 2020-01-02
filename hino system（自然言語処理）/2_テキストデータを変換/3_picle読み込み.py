import pickle

with open('txtresult.pickle', mode='rb') as f:
    wagahai_words = pickle.load(f)

print(wagahai_words)

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

tagged_documents = []
for i, sentence in enumerate(wagahai_words):
    tagged_documents.append(TaggedDocument(sentence, [i]))  # TaggedDocument型のオブジェクトをリストに格納

# size：分散表現の次元数
# window：対象単語を中心とした前後の単語数
# min_count：学習に使う単語の最低出現回数
# epochs:epochs数
# dm：学習モデル=DBOW（デフォルトはdm=1で、学習モデルはDM）
model = Doc2Vec(documents=tagged_documents,
                vector_size=100,
                min_count=5,
                window=5,
                epochs=20,
                dm=0)

print(wagahai_words[0])  # 最初の文章を表示
print(model.docvecs[0])  # 最初の文章のベクトル

print(model.docvecs.most_similar(0))

for p in model.docvecs.most_similar(0):
    print(wagahai_words[p[0]])