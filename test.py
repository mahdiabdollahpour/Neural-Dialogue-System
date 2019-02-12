import matplotlib.pyplot as plt
from lda import cluster_questions
from data_preprocessor import generate_data_from_clusters

#
# topic_nums = [50, 100, 150, 200, 250, 300]
# perps = []
# for t in topic_nums:
#     perp = cluster_questions(t, 'datasets/DialogQA/result' + str(t) + '/')
#     perps.append(perp)
#     print(perp)
#
# plt.plot(topic_nums, perps)
# plt.show()


generate_data_from_clusters('datasets/DialogQA/result300',
                            'datasets/DialogQA/pairs/src.txt', 'datasets/DialogQA/pairs/dest.txt')
