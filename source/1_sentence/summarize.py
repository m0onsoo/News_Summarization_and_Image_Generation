from transformers import pipeline
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from urllib.request import urlretrieve
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os
from datetime import datetime

current_directory = os.path.dirname(__file__)
# path = os.path.abspath(os.path.join(current_directory, "../data/"))

glove_dict = dict()
f = open(current_directory + '//glove.6B.100d.txt', encoding="utf8") # 100차원의 GloVe 벡터를 사용

for line in f:
    word_vector = line.split()
    word = word_vector[0]
    word_vector_arr = np.asarray(word_vector[1:], dtype='float32') # 100개의 값을 가지는 array로 변환
    glove_dict[word] = word_vector_arr
f.close()

# nltk에서 제공하는 불용어 받기
stop_words = stopwords.words('english')

# functions
# 토큰화 함수
def tokenization(sentences):
    return [word_tokenize(sentence) for sentence in sentences]

# 전처리 함수
def preprocess_sentence(sentence):
  # 영어를 제외한 숫자, 특수 문자 등은 전부 제거. 모든 알파벳은 소문자화
  sentence = [re.sub(r'[^a-zA-z\s]', '', word).lower() for word in sentence]

  # 불용어가 아니면서 단어가 실제로 존재해야 한다.
  return [word for word in sentence if word not in stop_words and word]

# 위 전처리 함수를 모든 문장에 대해서 수행. 이 함수를 호출하면 모든 행에 대해서 수행.
def preprocess_sentences(sentences):
    return [preprocess_sentence(sentence) for sentence in sentences]

# 단어 벡터의 평균으로부터 문장 벡터를 얻는다.
# 단, 불용어 제거해서 문장 길이가 0일 경우 100차원의 영벡터 리턴
# 현재 사용중인 GloVe 벡터의 차원은 100. 100차원 영벡터 만들기
embedding_dim = 100
zero_vector = np.zeros(embedding_dim)

def calculate_sentence_vector(sentence):
  if len(sentence) != 0:
    return sum([glove_dict.get(word, zero_vector) 
                  for word in sentence])/len(sentence)
  else:
    return zero_vector
  
  # 각 문장에 대해서 문장 벡터를 반환

def sentences_to_vectors(sentences):
    return [calculate_sentence_vector(sentence) 
              for sentence in sentences]

# 코사인 유사도를 구한 유사도 행렬 만들기
# 행렬의 크기는 (문장 개수 x 문장 개수)

def similarity_matrix(sentence_embedding):
  sim_mat = np.zeros([len(sentence_embedding), len(sentence_embedding)])
  for i in range(len(sentence_embedding)):
      for j in range(len(sentence_embedding)):
        sim_mat[i][j] = cosine_similarity(sentence_embedding[i].reshape(1, embedding_dim),
                                          sentence_embedding[j].reshape(1, embedding_dim))[0,0]
  return sim_mat

def calculate_score(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    try:
      scores = nx.pagerank(nx_graph, max_iter=200)
      return scores
    except:
      return None

def ranked_sentences(sentences, scores, n):
    if scores is not None:
      top_scores = sorted(((scores[i],s) 
                          for i,s in enumerate(sentences)), 
                                  reverse=True)
      top_n_sentences = [sentence 
                          for score,sentence in top_scores[:n]]
      return " ".join(top_n_sentences)
    else:
       return None

def news_to_summary(news_data):
    # html 태그 제거
    to_clean = re.compile("<.*?>")
    news_data = re.sub(to_clean, "", news_data)

    # 기사를 문장 단위로 쪼개줌
    sentences = sent_tokenize(news_data)

    # 문장을 단어 단위로 토큰화
    tokenized_sentences = tokenization(sentences)

    # 토큰을 소문자 등으로 전처리
    tokenized_sentences = preprocess_sentences(tokenized_sentences)

    # 문장을 GloVe 사용해 임베딩 벡터 만들기
    sentenceEmbedding = sentences_to_vectors(tokenized_sentences)

    # 문장 간 코사인 유사도를 나타낸 유사도 행렬 생성
    simMatrix = similarity_matrix(sentenceEmbedding)

    # pagerank 알고리즘 이용한 문장별 점수 계산
    score = calculate_score(simMatrix)

    # 상위 5개 문장을 최종 요약본으로 반환
    summary = ranked_sentences(sentences, score, 10)

    return summary

current_directory = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(current_directory, "../../data/"))
today = datetime.now().date()
data = pd.read_csv(path + f"//news_titles_{today}.csv")

news_data = list(data.loc[:, 'Contents'])

# BART 모델 로드
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summarized = []
for i, news in enumerate(news_data):
  print(f"{i}번째 문서 작업 중")
  print("Summarizing News")
  print("-----------------------------------------")
  print()

  try: 
    summarized_news = news_to_summary(news)
    summary = summarizer(summarized_news, min_length=30, max_length = 60, do_sample=False)
    summary_text = summary[0]['summary_text']
    summarized.append(summary_text)
  except:
    summarized.append(None)
print("End")
data['Summarized'] = summarized
data.to_csv(path + f"/news_titles_{today}.csv", index=False)