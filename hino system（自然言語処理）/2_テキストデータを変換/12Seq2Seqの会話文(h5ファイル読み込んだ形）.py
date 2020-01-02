import pickle

with open('kana_chars.pickle', mode='rb') as f:
    chars_list = pickle.load(f)
print(chars_list)


##文章のベクトル化
##文章をone-hot表現に変換する関数を設定します
import numpy as np

# インデックスと文字で辞書を作成
char_indices = {}
for i, char in enumerate(chars_list):
    char_indices[char] = i
indices_char = {}
for i, char in enumerate(chars_list):
    indices_char[i] = char
    
n_char = len(chars_list)
max_length_x = 128

# 文章をone-hot表現に変換する関数
def sentence_to_vector(sentence):
    vector = np.zeros((1, max_length_x, n_char), dtype=np.bool)
    for j, char in enumerate(sentence):
        vector[0][j][char_indices[char]] = 1
    return vector
    
    
##
##返答作成用の関数 encoderのモデル、およびdecoderのモデルを読み込み、返答を作成するための関数を設定します。 

from keras.models import load_model

encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')

def respond(input_data, beta=5):
    state_value = encoder_model.predict(input_data)
    y_decoder = np.zeros((1, 1, n_char))  # decoderの出力を格納する配列
    y_decoder[0][0][char_indices['\t']] = 1  # decoderの最初の入力はタブ。one-hot表現にする。

    respond_sentence = ""  # 返答の文字列
    while True:
        y, h = decoder_model.predict([y_decoder, state_value])
        p_power = y[0][0] ** beta  # 確率分布の調整
        next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power)) 
        next_char = indices_char[next_index]  # 次の文字
        
        if (next_char == "\n" or len(respond_sentence) >= max_length_x):
            break  # 次の文字が改行のとき、もしくは最大文字数を超えたときは終了
            
        respond_sentence += next_char
        y_decoder = np.zeros((1, 1, n_char))  # 次の時刻の入力
        y_decoder[0][0][next_index] = 1

        state_value = h  # 次の時刻の状態

    return respond_sentence
    
    



##対話生成の検証 コーパスに無い文章を使って、対話生成の検証を行います。

sentences = ["けんじさん、こんにちは。",
             "カムパネラってしってますか？",
             "ああ、おなかすいた。",
             "きょうはいいてんきですね。",
             "すきなたべものは、なんですか。",
             "きょうは、さんぽにいきました。",
             "パソコンかスマホは、つかったことありますか？"]

for sentence in sentences:
    vector = sentence_to_vector(sentence)
    print("Input:", sentence)
    print("Response", respond(vector))
    print()
    


##モデル同士の会話
response_a = "こんにちは。"
response_b = ""
for i in range(100):
    print("賢治A:", response_a)
    print()  
    vector_a = sentence_to_vector(response_a)
    
    response_b = respond(vector_a)
    print("賢治B:", response_b)
    print()
    vector_b = sentence_to_vector(response_b)
    
    response_a = respond(vector_b)