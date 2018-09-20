# -*- coding: utf-8 -*-

from PIL import Image
from skimage import transform,data
import numpy as np
from scipy.spatial.distance import pdist
import cv2
import os

class ImageRetrievor(object):
    def __init__(self, database_url):
        self.archives = []
        self.distances = []

        self.database_url = database_url

        self.std_width = 136 * 3
        self.std_height = 76 * 3

        self.retrieve_vectors = []

    def retrieve(self, imageUrl):
        pass

    def image_input(self, imageUrl):
        self.image_url = imageUrl
        self.image = Image.open(imageUrl)
        self.image_array = np.array(self.image)
        self.vector = transform.resize(self.image_array, (self.std_width, self.std_height)).flatten()


    def compute_distance(self, type='eu', p=3):
        for retrieve_vector in self.retrieve_vectors:
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
            elif (type=='SIFT'):
                realDis = .0
                for m in retrieve_vector:
                    realDis+=m[0].distance
                realDis /= len(retrieve_vector)
                self.distances.append(realDis)

    def compute_retrieve_vectors(self, type='Original'):
        if type=='Original':
            for retrieve_img_url in self.archives:
                retrieve_img = Image.open(retrieve_img_url)
                retrieve_vector = transform.resize(np.array(retrieve_img), self.image_array.shape).flatten()
                self.retrieve_vectors.append(retrieve_vector)
        if type=='SIFT':
            for retrieve_img_url in self.archives:
                kp_ret, des_ret = self.compute_features(retrieve_img_url)
                kp_or, des_or = self.compute_features(self.image_url)
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.knnMatch(des_ret, des_or, k = 1)
                self.retrieve_vectors.append(matches)

    def compute_features(self, imgUrl, type='SIFT'):
        img = cv2.imread(imgUrl)
        img = cv2.resize(img,(136 * 3,76 * 3))
        # cv2.imshow("original",img)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        if type=='SIFT':
            #使用SIFT
            sift = cv2.xfeatures2d.SIFT_create()
            self.keypoints, descriptor = sift.detectAndCompute(gray,None)
            # cv2.drawKeypoints(image = img,
            #                   outImage = img,
            #                   keypoints = self.keypoints,
            #                   flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            #                   color = (51,163,236))
            # cv2.imshow("SIFT",img)

        elif type=='SURF':
            surf = cv2.xfeatures2d.SURF_create()
            self.keypoints, descriptor = surf.detectAndCompute(gray,None)
            # cv2.drawKeypoints(image = img,
            #                   outImage = img,
            #                   keypoints = self.keypoints,
            #                   flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            #                   color = (51,163,236))
            # cv2.imshow("SURF",img)

        return self.keypoints, descriptor

    def compute_archives(self):
        for root ,dirs ,files in os.walk(self.database_url):
            for file in files:
                file_name = os.path.join(root,file)
                if file_name.split("\\")[-1].split('.')[-1] =='jpg':
                    self.archives.append(file_name)



