from ImageRetrievor import *
import cv2
from utils import *
import os

database_url = 'E://UserData\car_images'
retrieve_url = 'E://UserData\car_images\A0RP77\A0RP77_20151129151034_6765527322.jpg'

imageRetrievor = ImageRetrievor(database_url)
imageRetrievor.retrieve(retrieve_url)
