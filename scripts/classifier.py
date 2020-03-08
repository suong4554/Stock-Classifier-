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

smoothData = dataDict
#Smooths out the data
for data in dataDict:
    #window=90 because one stock quarter = 1/4th of a year
    smoothData[data]["open"]=(dataDict[data]["open"].rolling(window=90).mean()).dropna()
    smoothData[data]["close"]=(dataDict[data]["close"].rolling(window=90).mean()).dropna()
    smoothData[data]["high"]=(dataDict[data]["high"].rolling(window=90).mean()).dropna()
    smoothData[data]["low"]=(dataDict[data]["low"].rolling(window=90).mean()).dropna()
    smoothData[data]["volume"]=(dataDict[data]["volume"].rolling(window=90).mean()).dropna()
    smoothData[data]

print(smoothData[1].columns)

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.linear_model import LinearRegression


register_matplotlib_converters()

for data in dataDict:
    Y = smoothData[data]["close"].dropna().apply(lambda x: x*10000).astype("int")
    X = smoothData[data].drop("tdate", axis=1).drop("close", axis=1).drop("index", axis=1).drop("stock_id", axis=1).dropna().apply(lambda x: x*10000)





    feature_cols = list(X.columns)
    print(feature_cols)
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)
    #clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    #clf = clf.fit(x_train,y_train)
    reg = LinearRegression().fit(x_train, y_train)
    #Predict the response for test dataset
    #y_pred = clf.predict(x_test)
    y_pred = reg.predict(x_test)

    # Model Accuracy
    accuracySum = 0
    test = y_test.tolist()
    pred = y_pred.tolist()
    for i in range(len(test)):
        difference = abs(test[i] - pred[i])
        percent = difference/test[i]
        accuracySum +=percent

    accuracy = 1 - accuracySum/len(y_test)

    #print("Accuracy: " + str(data),metrics.accuracy_score(y_test, y_pred))
    print("Accuracy: " + str(data)," => ",accuracy)
    """

    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    import pydotplus
    import os

    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(home_dir + "\\img\\" + 'test'  + str(data) + '.png')
    Image(graph.create_png())

    print(len(y_train), len(y_pred))

    """
    plot_axis = smoothData[data].dropna()["tdate"]

    trainShape = len(y_train)
    y_train = np.true_divide(y_train, 10000)
    y_pred = np.true_divide(y_pred, 10000)
    y_test = np.true_divide(y_test, 10000)


    plt.title("Prediction Data: " + str(data))
    plt.plot(plot_axis[:trainShape],y_train,label=str(data) + ": Training")
    plt.plot(plot_axis[trainShape:],y_pred,label=str(data) + ": Predicted", color='red')
    plt.plot(plot_axis[trainShape:],y_test,label=str(data) + ": Actual", color='green')
    #dataDict[data]["close"].plot(label=str(data) + ": raw")
    plt.legend()
    plt.show()
#plt.title("Total Prediction Data Linear")
#plt.show()
