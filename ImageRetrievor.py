# -*- coding: utf-8 -*-

from PIL import Image
from skimage import transform,data
import numpy as np
from scipy.spatial.distance import pdist
import cv2
import os
from DictionaryTrainer import *

class ImageRetrievor(object):
    def __init__(self, database_url):
        self.archives = []
        self.distances = []

        self.database_url = database_url
        self.dictionary_trainer = DictionaryTrainer()

        self.std_width = 136 * 3
        self.std_height = 76 * 3

        self.retrieve_vectors = []

        self.min_distances = []
    def retrieve(self, imageUrl, type='BoW'):
        self.image_input(imageUrl, type)
        self.compute_archives()
        self.compute_retrieve_vectors(type)
        self.compute_distance(type)

        minIndex = self.distances.index(min(self.distances))
        file_name = self.archives[minIndex]
        img = cv2.imread(file_name)
        cv2.imshow("best_result",img)
        cv2.waitKey(0)

        self.min_distances = sorted(enumerate(self.distances), key=lambda x:x[1])

    def image_input(self, imageUrl, type):
        self.image_url = imageUrl
        if (type=='flatten'):
            self.image = Image.open(imageUrl)
            self.image_array = np.array(self.image)
            self.vector = transform.resize(self.image_array, (self.std_width, self.std_height)).flatten()
        elif (type=='BoW'):
            self.vector = None #Can only be computed after training process done.

    def compute_distance(self, type='BoW', p=3):
        index = 0
        for retrieve_vector in self.retrieve_vectors:
            index+=1
            print('computing distance:'+str(index)+'/'+str(len(self.retrieve_vectors)))
            if (type=='eu' or type=='BoW'):
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

    def compute_retrieve_vectors(self, type='BoW'):
        if type=='Original':
            for retrieve_img_url in self.archives:
                retrieve_img = Image.open(retrieve_img_url)
                retrieve_vector = transform.resize(np.array(retrieve_img), self.image_array.shape).flatten()
                self.retrieve_vectors.append(retrieve_vector)
        if type=='SIFT':
            index = 0
            for retrieve_img_url in self.archives:
                index+=1
                print('computing retrievector(SIFT):'+str(index)+'/'+str(len(self.archives)))
                kp_ret, des_ret = self.compute_features(retrieve_img_url)
                kp_or, des_or = self.compute_features(self.image_url)
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.knnMatch(des_ret, des_or, k = 1)
                self.retrieve_vectors.append(matches)
        if type=='BoW':
                self.dictionary_trainer.train(self.archives)
                self.vector = img_to_vect(self.image_url, self.dictionary_trainer.cluster_model)#Until now can we compute the vector of origin retrieve image
                img_bow_hist = self.dictionary_trainer.img_bow_hist
                for index in range(len(self.archives)):
                    self.retrieve_vectors.append(img_bow_hist[index])

    #A despercated method.
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
        print('computing archives...')
        for root ,dirs ,files in os.walk(self.database_url):
            for file in files:
                file_name = os.path.join(root,file)
                if file_name.split("\\")[-1].split('.')[-1] =='jpg' and file_name!=self.image_url:
                    self.archives.append(file_name)
        print('computing archives done')

    def computeRecallRate(self, return_N=10):
        carName = self.image_url.split('\\')[2]
        computeCorrectNum = 0

        root_Url = self.database_url+"\\"+carName
        #If exists no correct picture. Divide zero make it exception.
        actualCorrectNum = len([name for name in os.listdir(root_Url) if os.path.isfile(os.path.join(root_Url, name))])
        for index in range(return_N):
            #print("dected:"+self.archives[self.min_distances[index][0]])
            if self.archives[self.min_distances[index][0]].split('\\')[2]==carName:
                computeCorrectNum = computeCorrectNum+1

        #print('compute correct num:'+str(computeCorrectNum)+"/actual correct num"+str(actualCorrectNum))
        #print("RECALL RATE:"+str(computeCorrectNum/actualCorrectNum))
        return computeCorrectNum/actualCorrectNum

    def computePrecisonRate(self, return_N=10):
        carName = self.image_url.split('\\')[2]
        computeCorrectNum = 0

        root_Url = self.database_url+"\\"+carName
        for index in range(return_N):
            #print("dected:"+self.archives[self.min_distances[index][0]])
            if self.archives[self.min_distances[index][0]].split('\\')[2]==carName:
                computeCorrectNum = computeCorrectNum+1

        #print('compute correct num:'+str(computeCorrectNum)+"/return num"+str(return_N))
        #print("PRECISION RATE:"+str(computeCorrectNum/return_N))
        return computeCorrectNum/return_N
