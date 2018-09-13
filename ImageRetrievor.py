# -*- coding: utf-8 -*-

from PIL import Image
from skimage import transform,data
import numpy as np
from scipy.spatial.distance import pdist

class ImageRetrievor(object):
    def __init__(self, image_url):
        self.archives = ['2.jpg', '3.jpg']
        self.distances = []
        self.image_url = image_url

        self.image = Image.open(image_url)
        self.image_array = np.array(self.image)
        self.vector = self.image_array.flatten()

    def compute_distance(self, type='eu', p=3):
        for retrieve_img_url in self.archives:
            retrieve_img = Image.open(retrieve_img_url)
            retrieve_vector = transform.resize(np.array(retrieve_img), self.image_array.shape).flatten()

            if (type=='eu'):
                self.distances.append(np.linalg.norm(self.vector-retrieve_vector))
            elif (type=='min'):
                self.distances.append(pdist(np.vstack([self.vector, retrieve_vector]),'minkowski',p)[0])
            elif (type=='cos'):
                num = float(self.vector.dot(retrieve_vector))
                denom = np.linalg.norm(self.vector) * np.linalg.norm(retrieve_vector)
                cos = num / denom #余弦值
                sim = 0.5 + 0.5 * cos #归一化
                self.distances.append(sim)

