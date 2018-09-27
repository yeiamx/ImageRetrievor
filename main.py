from ImageRetrievor import *
import cv2
from utils import *
import os
from scipy.interpolate import spline
import matplotlib.pyplot as plt

database_url = 'E://UserData\car_images'
# retrieve_url = 'E://UserData\car_images\A1LV26\A1LV26_20151201141841_6777645514.jpg' #good
# retrieve_url = 'E://UserData\car_images\A2SN82\A2SN82_20151201190112_6779523485.jpg' #bad
retrieve_url = 'E://UserData\car_images\A1TA21\A1TA21_20151201195217_3091692917.jpg' #quite ok(soso)



imageRetrievor = ImageRetrievor(database_url)
imageRetrievor.retrieve(retrieve_url)

#imageRetrievor.drawRecallRate(return_N=5)
#imageRetrievor.drawPrecisonRate(return_N=5)
# for mindistance in mindistances:
#     print(imageRetrievor.archives[mindistance[0]])
#     img = cv2.imread(imageRetrievor.archives[mindistance[0]])
def drawRecallPrecisonCurve(imageRetrievor, maxN):
    power_precison = []
    power_recall = []

    for index in range(maxN):
        power_precison.append(imageRetrievor.computePrecisonRate(index+1))
        power_recall.append(imageRetrievor.computeRecallRate(index+1))
    power_precison = np.array(power_precison)
    power_recall = np.array(power_recall)

    T = np.array(range(maxN))
    xnew = np.linspace(T.min(),T.max(),300)
    power_smooth_precison = spline(T,power_precison,xnew)
    power_smooth_call = spline(T,power_recall,xnew)


    plt.plot(xnew,power_smooth_precison)
    plt.plot(xnew,power_smooth_call)
    plt.show()

maxN = 10
drawRecallPrecisonCurve(imageRetrievor, maxN)
for index in range(maxN):
    print("detected:"+imageRetrievor.archives[imageRetrievor.min_distances[index][0]])