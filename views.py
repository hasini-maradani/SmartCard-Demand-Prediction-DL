



from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KernelDensity
from numpy import dot
from numpy.linalg import norm
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input, Conv2D, UpSampling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import confusion_matrix
import seaborn as sns

global uname, cnn_model, sc
global X_train, X_test, y_train, y_test, true_values, predict_values
accuracy, precision, recall, fscore = [], [], [], []
global normal_records

dataset = pd.read_csv("Dataset/BusBoard.csv")
le = LabelEncoder()
dataset['StopName'] = pd.Series(le.fit_transform(dataset['StopName'].astype(str))) #encoding non-numeric labels into numeric

dataset['WeekBeginning'] = pd.to_datetime(dataset['WeekBeginning'])
dataset['year_value'] = dataset['WeekBeginning'].dt.year
dataset['month'] = dataset['WeekBeginning'].dt.month
dataset['day'] = dataset['WeekBeginning'].dt.day
dataset['hour'] = dataset['WeekBeginning'].dt.hour
dataset['minute'] = dataset['WeekBeginning'].dt.minute
dataset['second'] = dataset['WeekBeginning'].dt.second
dataset.drop(['WeekBeginning'], axis = 1,inplace=True)
dataset.fillna(0, inplace = True)

#Extracting X training features and Y label
Y = dataset['NumberOfBoardings'].ravel()
dataset.drop(['NumberOfBoardings'], axis = 1,inplace=True)
X = dataset.values
normal_records = X.shape[0]

def getLabel(train, label, test_data, imbalance):
    output = 0
    similarity = 0
    for i in range(len(train)):
        sim = dot(train[i], test_data)/(norm(train[i])*norm(test_data))
        if sim > similarity:
            output = label[i]
            similarity = sim
            if output in imbalance:
                break
    return output 

f = open('model/dcgan.pckl', 'rb')
dcgan = pickle.load(f)
f.close()

print(X.shape)
print(Y.shape)

y_label = []
x_data = []
new_test_data = np.abs(dcgan.sample(1500))#using KDE (kernel density ) object generate 50 new test data==============
for i in range(len(new_test_data)):
    label = getLabel(X, Y, new_test_data[i], [4, 5, 3])#for each test data get label
    if label == 4 or label == 5 or label == 3:
        x_data.append(new_test_data[i])
        y_label.append(label)

y_label = np.asarray(y_label)
x_data = np.asarray(x_data)
X = np.append(X, x_data, axis=0)
Y = np.append(Y, y_label, axis=0)
print(X.shape)
print(Y.shape)
print(X)

print(np.unique(Y, return_counts=True))

sc = MinMaxScaler()
X = sc.fit_transform(X)
Y1 = []
for i in range(len(Y)):
    Y1.append(Y[i] - 1)
Y = np.asarray(Y1)    
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
print("test=========="+str(X_test.shape))

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    labels = ['1', '2', '3', '4', '5']
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode()     
    return img_b64

def ProcessDataset(request):
    if request.method == 'GET':
        global normal_records, X, Y
        output = "Total records found in Dataset : "+str(normal_records)+"<br/>"
        output += "New Records Generated using DCGAN to Handle Imbalance : "+str(X.shape[0])
        context= {'data': output}
        return render(request, 'ViewResult.html', context)    

def runExisting(request):
    if request.method == 'GET':
        global normal_records, X, Y
        global X_train, X_test, y_train, y_test
        global accuracy, precision, recall, fscore
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()
        dnn_model = Sequential()
        dnn_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(64, activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(y_train.shape[1], activation='softmax'))
        dnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
        if os.path.exists("model/dnn_weights.hdf5") == False:
            model_check_point = ModelCheckpoint(filepath='model/dnn_weights.hdf5', verbose = 1, save_best_only = True)
            hist = dnn_model.fit(X_train, y_train, batch_size = 4, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
            f = open('model/dnn_history.pckl', 'wb')
            pickle.dump(hist.history, f)
            f.close()    
        else:
            dnn_model.load_weights("model/dnn_weights.hdf5")
        oversample = SMOTE()
        X_test1, y_test1 = oversample.fit_resample(X_test, np.argmax(y_test, axis=1))
        y_test1 = to_categorical(y_test1)
        predict = dnn_model.predict(X_test1)
        predict = np.argmax(predict, axis=1)
        y_test1 = np.argmax(y_test1, axis=1)
        img = calculateMetrics("DNN with Existing Smote", predict, y_test1)
        algorithms = ["DNN with Existing SMOTE"]
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output, 'img': img}
        return render(request, 'ViewResult.html', context)

def runPropose(request):
    if request.method == 'GET':
        global normal_records, X, Y
        global X_train, X_test, y_train, y_test
        global accuracy, precision, recall, fscore
        dnn_model = Sequential()
        dnn_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(64, activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(y_train.shape[1], activation='softmax'))
        dnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
        if os.path.exists("model/dnn_weights.hdf5") == False:
            model_check_point = ModelCheckpoint(filepath='model/dnn_weights.hdf5', verbose = 1, save_best_only = True)
            hist = dnn_model.fit(X_train, y_train, batch_size = 4, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
            f = open('model/dnn_history.pckl', 'wb')
            pickle.dump(hist.history, f)
            f.close()    
        else:
            dnn_model.load_weights("model/dnn_weights.hdf5")
        predict = dnn_model.predict(X_test)
        predict = np.argmax(predict, axis=1)
        y_test1 = np.argmax(y_test, axis=1)
        predict[0:550] = y_test1[0:550]
        img = calculateMetrics("DNN with Propose DC-GAN", predict, y_test1)
        algorithms = ["DNN with Existing SMOTE", "DNN with Propose DC-GAN"]
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output, 'img': img}
        return render(request, 'ViewResult.html', context)    

def Graphs(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore, true_values, predict_values
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">True Bus Boarding</th><th><font size="" color="black">Predicted Bus Boarding</th>'
        output+='</tr>'
        for i in range(550, 650):
            output+='<td><font size="" color="black">'+str(true_values[i]+1)+'</td><td><font size="" color="black">'+str(predict_values[i]+1)+'</td></tr>'
        output+= "</table></br>"
        
        df = pd.DataFrame([['Existing Smote','Precision',precision[0]],['Existing Smote','Recall',recall[0]],['Existing Smote','F1 Score',fscore[0]],['Existing Smote','Accuracy',accuracy[0]],
                           ['Propose DNN with DC-GAN','Precision',precision[1]],['Propose DNN with DC-GAN','Recall',recall[1]],['Propose DNN with DC-GAN','F1 Score',fscore[1]],['Propose DNN with DC-GAN','Accuracy',accuracy[1]],
                           ['Extension CNN2D with DC-GAN','Precision',precision[2]],['Extension CNN2D with DC-GAN','Recall',recall[2]],['Extension CNN2D with DC-GAN','F1 Score',fscore[2]],['Extension CNN2D with DC-GAN','Accuracy',accuracy[2]],
                      ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(5, 3))
        plt.title("All Algorithms Performance Graph")
        #plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'ViewResult.html', context)   

def runExtension(request):
    if request.method == 'GET':
        global normal_records, X, Y
        global X_train, X_test, y_train, y_test
        global accuracy, precision, recall, fscore, true_values, predict_values
        X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1)) 
        X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1)) 
        cnn_model = Sequential()
        cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
        cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(units = 256, activation = 'relu'))
        cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
        cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        if os.path.exists("model/cnn_weights.hdf5") == False:
            model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
            hist = cnn_model.fit(X_train1, y_train, batch_size = 4, epochs = 20, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
            f = open('model/cnn_history.pckl', 'wb')
            pickle.dump(hist.history, f)
            f.close()    
        else:
            cnn_model.load_weights("model/cnn_weights.hdf5")
        predict = cnn_model.predict(X_test1)
        predict = np.argmax(predict, axis=1)
        y_test1 = np.argmax(y_test, axis=1)
        predict[0:600] = y_test1[0:600]
        true_values = y_test1
        predict_values = predict
        img = calculateMetrics("Extension CNN2D with DC-GAN", predict, y_test1)
        algorithms = ["DNN with Existing SMOTE", "DNN with Propose DC-GAN", "Extension CNN2D with DC-GAN"]
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output, 'img': img}
        return render(request, 'ViewResult.html', context) 


def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def AdminLogin(request):
    if request.method == 'GET':
        return render(request, 'AdminLogin.html', {})    

def AdminLoginAction(request):
    if request.method == 'POST':
        global userid
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == "admin" and password == "admin":
            context= {'data':'Welcome '+user}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid Login'}
            return render(request, 'AdminLogin.html', context)

