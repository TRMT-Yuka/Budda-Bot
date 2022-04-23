# Budda-Bot

## 動作に必要なファイル
+ sbert_stairs
+ Buddha_dict.binaryfile
+ Yahoo_dict.binaryfile
+ requirements.txt
+ Buddha_QA_BERT.ipynb

## 動作の前に
requirements.txtに必要なモジュールが全て書かれています．（データ分析用のモジュールも含まれますので実際はもっと少ないです）

Buddha_QA_BERT.ipynb　内に書かれた以下のコードを順に実行してください．

```
# 初めに一度読み込んでください
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sbert = SentenceTransformer('./sbert_stair')
print("sbert OK!")

def similarity(id1, id2):
    return cosine_similarity([vectors[id2idx[id1]]], [vectors[id2idx[id2]]])[0][0]

with open('Butta_dict.binaryfile', 'rb') as bf:
    Butta_dict = pickle.load(bf)
    
with open('Yahoo_dict.binaryfile', 'rb') as yf:
    Yahoo_dict = pickle.load(yf)
```

上記は初めに各種モジュールやデータを読み込むためのコードです．


## Buddha-Botへの質問
以下のコードを実行し，質問文を入力してください．
仏陀の返答から最適なものを選び出力します．

```
#本セルを実行後．質問を入力すると回答が表示されます
print("【ブッダに聞いてみたいことは何ですか？】")
input_Q = input() 

Butta_cos = {}
for key in Butta_dict:
    Butta_cos[key] = cosine_similarity([Butta_dict[key]],[sbert.encode(input_Q)])
    
max_k = max(Butta_cos, key=Butta_cos.get) 
print("")
print("【お答え】")
print(max_k)

del Butta_cos
```


## Yahoo-Botへの質問
以下のコードを実行し，質問文を入寮してください．
Yahooの返答から最適なものを選び出力します．

```
print("【Yahooに聞いてみたいことは何ですか？】")
input_Q = input() 

Yahoo_cos = {}
for key in Yahoo_dict:
    Yahoo_cos[key] = cosine_similarity([Yahoo_dict[key]],[sbert.encode(input_Q)])
    
max_k = max(Yahoo_cos, key=Yahoo_cos.get) 
print("")
print("【お答え】")
print(max_k)

del Yahoo_cos
```

本Yahooボットは，ヤフーデータセット「Yahoo!知恵袋データ 第3版」を利用しています．
大カテゴリ「生き方と恋愛、人間関係の悩み」に属するQAの組をランダムに770取得し構築しています．
