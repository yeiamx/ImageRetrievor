from ImageRetrievor import *
import cv2
from utils import *
import os

database_url = 'E://UserData\car_images'
retrieve_url = 'E://UserData\car_images\A0RP77\A0RP77_20151128214453_3084006716.jpg'

imageRetrievor = ImageRetrievor(database_url)
imageRetrievor.retrieve(retrieve_url)

imageRetrievor.drawRecallRate()
# for mindistance in mindistances:
#     print(imageRetrievor.archives[mindistance[0]])
#     img = cv2.imread(imageRetrievor.archives[mindistance[0]])

