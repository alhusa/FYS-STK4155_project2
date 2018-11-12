import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import callbacks

class mlp:
    """
    Class to store the neural network
    """
    def __init__(self):
        """
        function to initialize the network
        """
        #set if the network prints the progress
        self.verbose = False
        #toggle the keras nettwork
        self.run_keras = True
        #learning rate
        self.eta = 0.1
        #momentum factor
        self.momentum = 0.7
        #number of hidden nodes
        self.hidden = 12
        #number of inputs
        self.ninput = 1600
        #number of outputs
        self.noutput = 2
        #size of minibatch
        self.minibatch = 20
        #number of epochs between checking earlystopping
        self.early = 10
        #weights for hidden layer
        self.v = (2/np.sqrt(self.ninput + 1)) * np.random.random_sample((self.ninput + 1 ,self.hidden)) -1/np.sqrt(self.ninput + 1)
        #store previous deltas for momentum
        self.vprev = np.zeros((self.ninput + 1 ,self.hidden))
        #weights for output layer
        self.w = (2/np.sqrt(self.hidden + 1))* np.random.random_sample((self.hidden + 1 ,self.noutput)) -1/np.sqrt(self.hidden + 1)
        #store previous deltas for momentum
        self.wprev = np.zeros((self.hidden + 1 ,self.noutput))


    def earlystopping(self, inputs, targets, valid, validtargets):
        """
        The earlystopping function runs the training function a number of epoch and evaluates the nn. The training
        is stopped if the network show no sign of improvement. A validation set is used to assess the model.
        Inputs: training data, training labels, validation data, validation labels.
        """

        #creats arrays to store MSE
        last_ac = np.zeros(10)
        timestart = 0
        overfit = 0
        for k in range(0,10000):

            #trains the MLP using all the test data
            for i in range(0,inputs.shape[0],self.minibatch):
                self.train(inputs[i:i+self.minibatch,:],targets[i:i+self.minibatch,:])



            #reorder the data so the program is not trained in the same way every epoch
            order = list(range(np.shape(inputs)[0]))
            np.random.shuffle(order)
            inputs = inputs[order,:]
            targets = targets[order,:]


            #checks the model with the validation set every 100 epochs
            if k%self.early == 0:
                timend =time.time() - timestart

                #get accuracy of the model using the validation set
                ac, conf =  mlp1.confusion(valid,validtargets)

                #stores the last 10 ac scores
                last_ac[1:] = last_ac[:9]
                last_ac[0] = ac
                diff = ac - np.mean(last_ac)

                if self.verbose:
                    #get accuracy of the model using the traing set
                    ac_train, conf_train =  mlp1.confusion(inputs,targets)
                    print("--------------------")
                    print("After %d epochs" %k)
                    print("Validation set accuracy: %.5f" %ac)
                    print("Training set accuracy: %.5f" %ac_train)


                #cheks if no improvements are found and increments overfit variable
                if diff <= 1e-10: overfit += 1
                #overfit varibale is reset if an improvement is found
                else:
                    overfit = 0
                    best_v = self.v
                    best_w = self.w

                #stops if there is little change in accuracy or
                if overfit > 5 or ac == 100:
                    self.v = best_v
                    self.w = best_w
                    break
        return k

    def train(self, inputs, targets):
        """
        Trains the network by running it forwards and then using backpropagation to adjust the weights.
        Inputs: training data, training labels.
        Outputs: Number of runs before earlystopping and array of accuracy scores.
        """
        #create array to store errors
        errOm = np.zeros((self.noutput,1))
        errOmsum = np.zeros((self.noutput,1))
        errHm = np.zeros((self.hidden,1))


        #runs the program forward
        outputH, outputO = self.forward(inputs)

        #calculate output error
        errOm = (outputO.T - targets) * outputO.T * np.subtract(1, outputO.T)


        #calculate hidden layer error
        summerrm = self.w[1:,:] @ errOm.T
        errHm =  (outputH * np.subtract(1, outputH))  * summerrm

        #put the bias in the hidden layer output
        outputHmi = np.zeros((self.hidden + 1,self.minibatch))
        outputHmi[0,:] = -1
        outputHmi[1:,:] = outputH

        #adjusting weight for the output
        deltaw = self.eta * outputHmi @ (errOm/self.minibatch)
        self.w = self.w -  deltaw + self.momentum * self.wprev
        self.wprev = deltaw

        #put the bias in the inputs
        inputsi = np.zeros((inputs.shape[1] + 1,self.minibatch))
        inputsi[0,:] = -1
        inputsi[1:,:] = inputs.T

        #adjusting wheight for hidden layer
        deltav = self.eta * inputsi @ (errHm/self.minibatch).T

        self.v = self.v - deltav + self.momentum * self.vprev
        self.vprev = deltav

    #function to run the MPL forward
    def forward(self, inputs):
        """
        Runs the network forwards.
        Inputs: training data.
        Outputs: Hiddel layer outputs and network output.
        """
        #puts bias in the inputs
        inputss = np.zeros((inputs.shape[0],inputs.shape[1] + 1))
        inputss[:,0] = -1
        inputss[:,1:] = inputs

        #caculate the hidden layer output
        outputhm = self.v.T @ inputss.T

        outputhm = self.activate(outputhm)

        #puts bias in the hiddel layer outputs
        outputhmi = np.zeros((self.hidden + 1,inputs.shape[0]))
        outputhmi[0,:] = -1
        outputhmi[1:,:] = outputhm

        #calculate the outputs of the MLP
        outputom = self.w.T @ outputhmi
        outputom = self.activate(outputom)

        #returns the hidden layer output and the output output
        return outputhm, outputom

    def confusion(self, inputs, targets):
        """
        Calculates the accuracy score and confusion matrix for the network.
        Inputs: Data and labels that are going to be assessed.
        Outputs: accuracy score and confusion matrix.
        """
        #create confuison matrix
        conf = np.zeros((self.noutput,self.noutput))

        #creates a value to store the number of correct classifications
        correct = 0


        #runs the moodel forwards
        outputHi, outputOu = self.forward(inputs)

        #finds the target result and the estimated resilts
        tarind = np.argmax(targets,axis=1)
        estind = np.argmax(outputOu,axis=0)

        #for loop runs through the test input
        for i in range(0,inputs.shape[0]):
            #puts increments the values in the confusion matrix
            #based on the results
            conf[tarind[i]][estind[i]] = conf[tarind[i]][estind[i]] + 1

            #if a value is placed in the diag then classification is correct
            if tarind[i] == estind[i]:
                correct = correct + 1

        #gets the percentage of correct classifications
        percor = correct/inputs.shape[0]

        return percor * 100, conf

    def activate(self, inputs):
        """
        Calculates the activation function for an output.
        Inputs: outputs of a layer in the network
        Outputs: the activation function result.
        """
        return 1 / (1 + ( np.exp( - inputs)))

L = 40
label_filename = 'Ising2DFM_reSample_L40_T=All_labels.pkl'
dat_filename = 'Ising2DFM_reSample_L40_T=All.pkl'

# Read in the labels
with open(label_filename, "rb") as f:
    labels = pickle.load(f)

# Read in the corresponding configurations
with open(dat_filename, "rb") as f:
    data = np.unpackbits(pickle.load(f)).reshape(-1, 1600).astype("int")

# Set spin-down to -1
data[data == 0] = -1

#generate onehot vector
target = np.zeros((np.shape(data)[0],2));
for x in range(0,2):
    indices = np.where(labels==x)
    target[indices,x] = 1


# Set up slices of the dataset
ordered = slice(0, 70000)
critical = slice(70000, 100000)
disordered = slice(100000, 160000)

#citical data
critical_data = data[critical]
critical_label = target[critical]

#ordered and disordered data
datawo = np.concatenate((data[ordered], data[disordered]))
labelswo = np.concatenate((target[ordered], target[disordered]))



#randomly shuffle the data
order = list(range(np.shape(datawo)[0]))
np.random.shuffle(order)
datawo = datawo[order,:]
labelswo = labelswo[order]


# Split data into k sets
foldsm = []
foldst = []
for i in range(0,13000,1300):
    foldsm.append(datawo[i:i+1300,:])
    foldst.append(labelswo[i:i+1300,:])



#arrays to store data
test_ac = np.zeros(10)
train_ac = np.zeros(10)
citical_ac = np.zeros(10)
k = np.zeros(10)
ac_test_keras = np.zeros(10)
ac_train_keras = np.zeros(10)
ac_critical_keras = np.zeros(10)

for i in range(0,10):

    # Test data is used to evaluate how good the completely trained network is.
    test = foldsm[i]
    test_targets = foldst[i]
    sum = 0

    valind = np.random.randint(9)
    if valind >= i:
        valind = valind + 1

    # Validation checks how well the network is performing and when to stop
    valid = foldsm[valind]
    valid_targets = foldst[valind]

    sumind = 0
    for j in range(0,10):
        if j != i and j != valind: sumind = sumind + foldsm[j].shape[0]

    #Training data to train the network
    train = np.zeros((sumind,1600))
    train_targets = np.zeros((sumind,2))
    placedind = 0
    for j in range(0,10):
        if j != i and j != valind:
            train[placedind:placedind+foldsm[j].shape[0],:] = foldsm[j]
            train_targets[placedind:placedind+foldsm[j].shape[0],:] = foldst[j]
            placedind = placedind + foldsm[j].shape[0]


    # initialize the network
    mlp1 = mlp()


    #run the mlp
    k[i]=mlp1.earlystopping(train, train_targets, valid, valid_targets)


    #array to store accuracy scores
    test_ac[i], mat = mlp1.confusion(test,test_targets)
    train_ac[i], train_mat = mlp1.confusion(train,train_targets)
    citical_ac[i], citical_mat = mlp1.confusion(critical_data,critical_label)


    if mlp1.run_keras:
        #using kera for comparison
        model = Sequential()
        model.add(Dense(mlp1.hidden, input_dim=mlp1.ninput, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        earlystop = callbacks.EarlyStopping(monitor='val_acc', min_delta=1e-7, patience=20, verbose=0, mode='auto')
        model.fit(train, train_targets, epochs=1000, verbose=0, batch_size=100, validation_data=(valid,valid_targets), callbacks=[earlystop])
        keras_pred_test = model.predict(test)
        keras_pred_train = model.predict(train)
        keras_pred_critical = model.predict(critical_data)



        #define values to be one or the other class
        keras_pred_test[keras_pred_test < 0.5] = 0
        keras_pred_test[keras_pred_test >= 0.5] = 1

        keras_pred_train[keras_pred_train < 0.5] = 0
        keras_pred_train[keras_pred_train >= 0.5] = 1

        keras_pred_critical[keras_pred_critical < 0.5] = 0
        keras_pred_critical[keras_pred_critical >= 0.5] = 1


        ac_test_keras[i] = metrics.accuracy_score(test_targets,keras_pred_test) * 100
        ac_train_keras[i] = metrics.accuracy_score(train_targets,keras_pred_train) * 100
        ac_critical_keras[i] = metrics.accuracy_score(critical_label,keras_pred_critical) * 100

    #print results for each fold
    print("--------------------------------------------------------\n")
    print("Own MLP:")
    print("Train set accuracy: %.4f%% Test set accuracy: %.4f%% Critical set accuracy: %.4f%%\n" %(train_ac[i], test_ac[i], citical_ac[i]))

    if mlp1.run_keras:
        print("keras MLP:")
        print("Train set accuracy: %.4f%% Test set accuracy: %.4f%% Critical set accuracy: %.4f%%\n" %(ac_train_keras[i], ac_test_keras[i], ac_critical_keras[i]))

    print("--------------------------------------------------------")

#print results
print("Created MLP:")
print("Average accuracy train data: %.2f%%. Min test: %.2f%%. Max test: %.2f%%. Test std: %.2f%%"
%(np.mean(train_ac), np.min(train_ac), np.max(train_ac), np.std(train_ac)))
print("Average accuracy test data: %.2f%%. Min test: %.2f%%. Max test: %.2f%%. Test std: %.2f%%"
%(np.mean(test_ac), np.min(test_ac), np.max(test_ac), np.std(test_ac)))
print("Average accuracy on critical data %.2f%%. Min test: %.2f%%. Max test: %.2f%%. Test std: %.2f%%"
%(np.mean(citical_ac), np.min(citical_ac), np.max(citical_ac), np.std(citical_ac)))


if mlp1.run_keras:
    print("keras MLP:")
    print("Average accuracy train data: %.2f%%. Min test: %.2f%%. Max test: %.2f%%. Test std: %.2f%%"
    %(np.mean(ac_train_keras), np.min(ac_train_keras), np.max(ac_train_keras), np.std(ac_train_keras)))
    print("Average accuracy test data: %.2f%%. Min test: %.2f%%. Max test: %.2f%%. Test std: %.2f%%"
    %(np.mean(ac_test_keras), np.min(ac_test_keras), np.max(ac_test_keras), np.std(ac_test_keras)))
    print("Average accuracy on critical data %.2f%%. Min test: %.2f%%. Max test: %.2f%%. Test std: %.2f%%"
    %(np.mean(ac_critical_keras), np.min(ac_critical_keras), np.max(ac_critical_keras), np.std(ac_critical_keras)))


#plot the accuracy scores
plt.plot(train_ac, 'b',label='Created MLP train')
plt.plot(test_ac,'--b',label='Created MLP test')
plt.plot(citical_ac,'-.b',label='Created MLP critical')
if mlp1.run_keras:
    plt.plot(ac_train_keras,'r',label='keras MLP train')
    plt.plot(ac_test_keras,'--r',label='keras MLP test')
    plt.plot(ac_critical_keras,'-.r',label='keras MLP critical')


if mlp1.run_keras:
    plt.title("Accuracy scores for test, training and critical data for the created and the keras MLP during a 10-fold", fontsize = 16)
else:
    plt.title("Accuracy scores for test, training and critical data for the created MLP during a 10-fold", fontsize = 16)
plt.legend(fontsize=16)
plt.xlabel('Fold number',fontsize=15)
plt.ylabel('Accuracy score [%]',fontsize=15)
plt.tick_params(labelsize=15)


plt.show()
