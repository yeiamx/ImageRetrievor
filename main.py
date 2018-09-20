from ImageRetrievor import *
import cv2
from utils import *
import os

type = 'SIFT'
database_url = 'E://UserData\car_images'

imageRetrievor = ImageRetrievor('1.jpg', database_url)
imageRetrievor.compute_archives()
# imageRetrievor.compute_archives()
# imageRetrievor.compute_retrieve_vectors(type=type)
# #imageRetrievor.compute_distance(type='min')
# imageRetrievor.compute_distance(type=type)

#print(imageRetrievor.distances)
print(imageRetrievor.archives)
