import os
from utils import *
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class DictionaryTrainer(object):
    def __init__(self):
        self.K = 500 #要聚类的数量，即字典的大小(包含的单词数)
        self.cluster_model = MiniBatchKMeans(n_clusters=self.K, init_size=3*self.K)

    def train(self, archives, option='compute'):
        print('starting training...')
        self.archives = archives
        img_descs = gen_all_surf_features(self.archives)
        if (option=='compute'):
            self.img_bow_hist = self.cluster_features(img_descs)
            if not os.path.exists('img_bow_hist.npy'):
                np.save('img_bow_hist.npy', self.img_bow_hist)
        elif (option=='load'):
            if os.path.exists('img_bow_hist.npy'):
                self.img_bow_hist = np.load('img_bow_hist.npy')
            else:
                print('there isnt exist any hist.You may want retrain the dataset using "compute" option.')
                exit(0)
        print('training done!')

    def cluster_features(self, img_descs):
        n_clusters = self.cluster_model.n_clusters #要聚类的种类数
        #将所有的特征排列成N*D的形式，其中N表示特征数，
        #D表示特征维度，这里特征维度D=64
        train_descs = [desc for desc_list in img_descs
                           for desc in desc_list]
        train_descs = np.array(train_descs)#转换为numpy的格式

        #判断D是否为64
        if train_descs.shape[1] != 64:
            raise ValueError('期望的SURF特征维度应为64, 实际为'
                             , train_descs.shape[1])
        #训练聚类模型，得到n_clusters个word的字典
        self.cluster_model.fit(train_descs)
        #raw_words是每张图片的SURF特征向量集合，
        #对每个特征向量得到字典距离最近的word
        img_clustered_words = [self.cluster_model.predict(raw_words)
                               for raw_words in img_descs]
        #对每张图得到word数目条形图(即字典中每个word的数量)
        #即得到我们最终需要的特征
        img_bow_hist = np.array(
            [np.bincount(clustered_words, minlength=n_clusters)
             for clustered_words in img_clustered_words])

        return img_bow_hist

