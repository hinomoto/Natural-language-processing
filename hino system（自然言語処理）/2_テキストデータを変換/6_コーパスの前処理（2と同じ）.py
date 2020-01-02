import re
import pickle
from janome.tokenizer import Tokenizer

with open("txtresult.txt", mode="r", encoding="utf-8") as f:  # ファイルの読み込み
    wagahai_original = f.read()

wagahai = re.sub("《[^》]+》", "", wagahai_original) # ルビの削除
wagahai = re.sub("［[^］]+］", "", wagahai) # 読みの注意の削除
wagahai = re.sub("[｜ 　「」\n]", "", wagahai) # | と全角半角スペース、「」と改行の削除

seperator = "。"  # 。をセパレータに指定
wagahai_list = wagahai.split(seperator)  # セパレーターを使って文章をリストに分割する
wagahai_list.pop() # 最後の要素は空の文字列になるので、削除
wagahai_list = [x+seperator for x in wagahai_list]  # 文章の最後に。を追加
        
t = Tokenizer()

wagahai_words = []
for sentence in wagahai_list:
    wagahai_words.append(t.tokenize(sentence, wakati=True))   # 文章ごとに単語に分割し、リストに格納
    
with open('txtresult2.pickle', mode='wb') as f:  # pickleに保存
    pickle.dump(wagahai_words, f)
    
from gensim.models import word2vec

# size : 中間層のニューロン数
# min_count : この値以下の出現回数の単語を無視
# window : 対象単語を中心とした前後の単語数
# iter : epochs数
# sg : skip-gramを使うかどうか 0:CBOW 1:skip-gram
model = word2vec.Word2Vec(wagahai_words,
                          size=100,
                          min_count=5,
                          window=5,
                          iter=20,
                          sg = 0)


print(model.wv.vectors.shape)  # 分散表現の形状
print(model.wv.vectors)  # 分散表現


print(len(model.wv.index2word))  # 語彙の数
print(model.wv.index2word[:10])  # 最初の10単語


print(model.wv.vectors[0])  # 最初のベクトル
print(model.wv.__getitem__("の"))  # 最初の単語「の」のベクトル