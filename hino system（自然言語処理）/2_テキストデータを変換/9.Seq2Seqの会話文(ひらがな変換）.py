import re

#novels = ["gingatetsudono_yoru.txt", "serohikino_goshu.txt", "chumonno_oi_ryoriten.txt",
#         "gusukobudorino_denki.txt", "kaeruno_gomugutsu.txt", "kaino_hi.txt", "kashiwabayashino_yoru.txt",
#         "kazeno_matasaburo.txt", "kiirono_tomato.txt", "oinomorito_zarumori.txt"]  # 青空文庫より
         
novels = ["txtresult.txt"]  # 青空文庫より
         


text = ""
for novel in novels:
    with open("hino_novels/"+novel, mode="r", encoding="utf-8") as f:  # ファイルの読み込み
        text_novel = f.read()
    text_novel = re.sub("《[^》]+》", "", text_novel)  # ルビの削除
    text_novel = re.sub("［[^］]+］", "", text_novel)  # 読みの注意の削除
    text_novel = re.sub("〔[^〕]+〕", "", text_novel)  # 読みの注意の削除
    text_novel = re.sub("[ 　\n「」『』（）｜※＊…]", "", text_novel)  # 全角半角スペース、改行、その他記号の削除
    text += text_novel

print("文字数:", len(text))
print(text)


#漢字をひらがなに変換
from pykakasi import kakasi

seperator = "。"  # 。をセパレータに指定
sentence_list = text.split(seperator)  # セパレーターを使って文章をリストに分割する
sentence_list.pop() # 最後の要素は空の文字列になるので、削除
sentence_list = [x+seperator for x in sentence_list]  # 文章の最後に。を追加

kakasi = kakasi()
kakasi.setMode("J", "H")  # J(漢字) からH(ひらがな)へ
conv = kakasi.getConverter()
for sentence in sentence_list:
    print(sentence)
    print(conv.do(sentence))
    print()
    
    
    

##################pykakasiの辞書に無い「苹」という文字が問題となる場合などは以下の処理を入れる
    
text = text.replace("苹果", "ひょうか")

seperator = "。"
sentence_list = text.split(seperator) 
sentence_list.pop() 
sentence_list = [x+seperator for x in sentence_list]

for sentence in sentence_list:
    print(sentence)
    print(conv.do(sentence))
    print()
##################



kana_text = conv.do(text)  # 全体をひらがなに変換
print(set(kana_text))  # set()で文字の重複をなくす


#テキストデータを保存し、いつでも使えるようにします。
print(kana_text)
with open("kana_kenji.txt", mode="w", encoding="utf-8") as f:
    f.write(kana_text)