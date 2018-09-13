from ImageRetrievor import *

imageRetrievor = ImageRetrievor('1.jpg')
imageRetrievor.compute_distance(type='min')
print(imageRetrievor.distances)