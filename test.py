from data_preprocessor import parse_DialogQA
# parse_DialogQA()
import matplotlib.pyplot as plt
from lda import cluster_questions

# preps = []
# topic_nums = [50, 100, 150, 200, 250, 300]
#
# for i in topic_nums:
#     preps.append(cluster_questions(300, 'datasets/DialogQA/result/'))
# print(preps)
# plt.plot(topic_nums, [-79.99301878152993, -79.97556660470399, -80.007857856378, -79.99956658415012, -80.0164515180253,
#                       -79.93820415860445])
# plt.show()



cluster_questions(250,'datasets/DialogQA/result/')


