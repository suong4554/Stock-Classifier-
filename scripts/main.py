import os
import pandas as pd
import numpy as np
from math import sqrt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#Visualization
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from pandas.plotting import register_matplotlib_converters



def visualize(test_y, pred_y, title):
    plt.scatter(test_y, pred_y)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title(title)
    plt.show()


# returns the data from the Exel table:
def load_df(dir_path, file_name):
    file = dir_path + "data\\" + file_name
    data = pd.read_excel(file, sheet_name="stock")
    return data

def encodeArr(train_df):
    for i in train_df:
        #Casting type to category for efficiency
        train_df[i] = train_df[i].fillna(train_df[i]).astype('category')
        #Built in python to convert each value in a column to a number
        train_df[i] = train_df[i].cat.codes
    return train_df

##########################################################################################
#####################################DATA LOADING###################################
# load the training data frame:
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
dataDF = load_df(home_dir, "stock-all.xls")

#Scales values
#dataDF["open"] = preprocessing.scale(dataDF["open"])
#dataDF["close"] = preprocessing.scale(dataDF["close"])

#Creates a dict of the data based on 'stock_id'
dataDict = dict(tuple(dataDF.groupby('stock_id')))

#resets indexes back to 0
for data in dataDict:
    dataDict[data] = dataDict[data].reset_index()

#Setting plot size
mpl.rc('figure', figsize=(15, 8))
# Adds lines to graph
style.use('ggplot')
########################################################################################
#####################################DATA PRE-PROCESSING###################################

#Plots Raw Data
"""
plt.title("Raw Data")
register_matplotlib_converters()
for data in dataDict:
    plt.plot(dataDict[data]["tdate"],dataDict[data]["close"],label=str(data) + ": Closing")
    #dataDict[data]["close"].plot(label=str(data) + ": raw")
plt.legend()
plt.show()
"""


closeData = {}
#Smooths out the data
for data in dataDict:
    #window=90 because one stock quarter = 1/4th of a year
    closeData[data]=(dataDict[data]["close"].rolling(window=90).mean())

#Plots total raw Data
"""
#adds dates
plt.title("Total Return")
register_matplotlib_converters()
for data in closeData:
    plt.plot(dataDict[data]["tdate"],closeData[data],label=str(data) + ": average")
    #dataDict[data]["close"].plot(label=str(data) + ": raw")
plt.legend()
#plt.show()
"""

#Calculates data returns based on first price
returnData = closeData
for data in dataDict:
    returnData[data] = returnData[data]/dataDict[data]["close"][0]
plt.clf()
#Plots percent returns based on first price
plt.title("Percent Return w/o outlier")
for data in returnData:
    #Gets rid of outlier data
    if data != 857:
        plt.plot(dataDict[data]["tdate"],returnData[data],label=str(data) + ": average")
    #dataDict[data]["close"].plot(label=str(data) + ": raw")
plt.legend()
#plt.show()


"""
dfData = pd.DataFrame.from_dict(returnData)
plt.imshow(dfData, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(dfData)), dfData.columns)
plt.yticks(range(len(dfData)), dfData.columns)
plt.show()
"""

###############################################KMeans######################################################
import pandas as pd
from sklearn.cluster import KMeans

#Find the Within Cluster Sum of Squared Errors
print("Fidning Sum of Squared Errors")
plt.clf()
plt.title("Elbow Curve")
for data in returnData:
    if data != 857:
        ret_var = pd.concat([dataDict[data]["tdate"].astype(np.int64), returnData[data]], axis=1).dropna()
        ret_var.columns  = ["Date", "Return"]
        X =  ret_var.to_numpy() #Converting ret_var into nummpy array

        sse = []
        for k in range(2,15):
            kmeans = KMeans(n_clusters = k)
            kmeans.fit(X)
            sse.append(kmeans.inertia_) #SSE for each n_clusterspl.plot(range(2,15), sse) (sum squared errors)
        plt.plot(range(2,15), sse, label=str(data) + ": sse")

plt.legend()
#plt.show()
#Chooses 5 Clusters as error rate becomes considerably smaller after 5

#Do the K-means magic
plt.clf()
plt.title("KMeans Plot Total Data")
print("Plotting Kmeans")
for data in returnData:
    if(data !=857):
        #plt.title("KMeans Plot: " + str(data))
        ret_var = pd.concat([dataDict[data]["tdate"].astype(np.int64), returnData[data]], axis=1).dropna()
        ret_var.columns  = ["Date", "Return"]
        X =  ret_var.to_numpy()
        kmeans = KMeans(n_clusters=5).fit(X)
        centroids = kmeans.cluster_centers_
        plt.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="Accent", s=1)
        #plt.scatter(centroids[:,0],centroids[:,1] , s=20)
plt.legend()
#plt.show()
