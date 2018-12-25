from gensim.models import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import numpy as np


def cluster_questions(topic_num, res_path, q_path='datasets\DialogQA\Qall.txt', a_path='datasets\DialogQA\Aall.txt'):
    with open(a_path, 'r', encoding='utf-8') as f:
        common_texts = [text.split() for text in f.readlines()]

    with open(q_path, 'r', encoding='utf-8') as f:
        questions = [text for text in f.readlines()]

    common_dictionary = Dictionary(common_texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    lda = LdaModel(common_corpus, num_topics=topic_num)

    questions_clusterd = [[] for i in range(topic_num)]
    print('Questions : ',len(questions))
    perp = lda.log_perplexity(common_corpus)
    for i, q in enumerate(questions):
        other_corpus = [common_dictionary.doc2bow(common_texts[i])]
        vector = lda[other_corpus]
        # print(vector[0])
        max_prob = 0
        for (idx, prob) in vector[0]:
            # print(idx)
            if prob > max_prob:
                topic = idx
                max_prob = prob
        questions_clusterd[topic].append(q)
        # print(topic)
    for top in range(topic_num):
        with open(res_path + str(top) + '.txt', 'w', encoding='utf-8') as f:
            for quest in questions_clusterd[top]:
                f.write(quest)
                # f.write('\n')

    return perp
