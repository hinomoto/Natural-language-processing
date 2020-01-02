with open("txtresult.txt", mode="r", encoding="utf-8") as f:  # ファイルの読み込み
    wagahai_original = f.read()

print(wagahai_original)

import re
import pickle

wagahai = re.sub("《[^》]+》", "", wagahai_original) # ルビの削除
wagahai = re.sub("［[^］]+］", "", wagahai) # 読みの注意の削除
wagahai = re.sub("[｜ 　「」\n]", "", wagahai) # | と全角半角スペース、「」と改行の削除

seperator = "。"  # 。をセパレータに指定
wagahai_list = wagahai.split(seperator)  # セパレーターを使って文章をリストに分割する
wagahai_list.pop() # 最後の要素は空の文字列になるので、削除
wagahai_list = [x+seperator for x in wagahai_list]  # 文章の最後に。を追加
        
print(wagahai_list)

with open('txtresult.pickle', mode='wb') as f:  # pickleに保存
    pickle.dump(wagahai_list, f)