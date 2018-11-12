import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn.metrics as met


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
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
        self.eta = 0.01
        #momentum factor
        self.momentum = 0.5
        #number of hidden nodes
        self.hidden = 20
        #number of inputs
        self.ninput = 1600
        #number of outputs
        self.noutput = 1
        #size of minibatch
        self.minibatch = 50
        #number of epochs between checking earlystopping
        self.early = 10
        #weights for hidden layer
        self.v = (2/np.sqrt(self.ninput + 1)) * np.random.random_sample((self.ninput + 1 ,self.hidden)) -1/np.sqrt(self.ninput + 1)
        #store previous deltas for momentum
        self.vprev = np.zeros((self.ninput + 1 ,self.hidden))
        #weights for output layer
        self.w = (2/np.sqrt(self.hidden + 1)) * np.random.random_sample((self.hidden + 1 ,self.noutput)) -1/np.sqrt(self.hidden + 1)
        #store previous deltas for momentum
        self.wprev = np.zeros((self.hidden + 1 ,self.noutput))


    def earlystopping(self, inputs, targets, valid, validtargets):
        """
        The earlystopping function runs the training function a number of epoch and evaluates the nn. The training
        is stopped if the network show no sign of improvement. A validation set is used to assess the model.
        Inputs: training data, training labels, validation data, validation labels.
        Outputs: Number of runs before earlystopping and array of accuracy scores.
        """

        #creats arrays to store MSE
        last_MSE = np.zeros(10)
        timestart = 0
        overfit = 0

        for k in range(0,10000):

            #trains the program for using all the test data
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

                #gets the MSE and R2 score for the validation set
                MSE, R2 =  mlp1.score(valid,validtargets)
                #stores the last 10 ac scores
                last_MSE[1:] = last_MSE[:9]
                last_MSE[0] = MSE
                diff = MSE - np.mean(last_MSE)


                timestart = time.time()

                #prints training and validation results if verbose is true
                if self.verbose:
                    MSE_train, R2_train =  mlp1.score(inputs,targets)
                    print("--------------------" )
                    print("After %d epochs"%k)
                    print("Validation set:")
                    print("MSE: %.5f    R2: %.5f" %(MSE, R2))
                    print("Training set:")
                    print("MSE: %.5f    R2: %.5f" %(MSE_train, R2_train))


                #cheks if no improvements are found and increments overfit variable
                if diff > 1e-10: overfit += 1
                #overfit varibale is reset if an improvement is found
                else:
                    overfit = 0
                    best_v = self.v
                    best_w = self.w

                #stops if there is little change in MSE
                if (overfit > 5 or MSE < 1e-3) and k>10 * self.early:
                    self.v = best_v
                    self.w = best_w
                    break


        return k

    def train(self, inputs, targets):
        """
        Trains the network by running it forwards and then using backpropagation to adjust the weights.
        Inputs: training data, training labels.
        """

        #runs the program forward
        outputH, outputO = self.forward(inputs)

        #calculate output error
        errOm = (outputO.T - targets)

        #calculate hidden layer error
        summerrm = self.w[1:,:] @ errOm.T

        errHm =  (outputH * np.subtract(1, outputH)) * summerrm

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

        #returns the hidden layer output and the output output
        return outputhm, outputom

    def score(self, inputs, targets):
        """
        Calculates the MSE and R2 scores of the network.
        Inputs: Data and labels that are going to be assessed.
        Outputs: MSE- and R2-scores.
        """
        #runs the moodel forward
        outputHi, outputOu = self.forward(inputs)

        target_e = 0 #target error
        target_a = 0 #target sum of target - target_avrage
        target_avg = np.mean(targets) #calcute the mean of targets
        n = inputs.shape[0]

        #sum of error
        target_e = np.sum((outputOu.T - targets)**2)
        target_a = np.sum((targets - target_avg)**2)

        MSE = target_e/(n) #calcute MSE
        R2 = 1 - (target_e/target_a) #calcute R2

        return MSE ,R2

    def activate(self, inputs):
        """
        Calculates the activation function for an output.
        Inputs: outputs of a layer in the network
        Outputs: the activation function result.
        """

        return 1 / (1 + ( np.exp( - inputs)))

def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0

    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E


sampn = 1000 #number of samples
nlevel = 0 #noise level

N0 = np.random.normal(0,nlevel, sampn)
L = 40

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(sampn,L))

#gets the target energies using the ising_energies function
target = ising_energies(states,L) + N0
target = target.reshape(sampn,1)

#get data for X matrix
states=np.einsum('...i,...j->...ij', states, states)
states=states.reshape((sampn,L*L))


# Split data into k sets
foldsm = []
foldst = []
for i in range(0,sampn,int(sampn/10)):
    foldsm.append(states[i:i+int(sampn/10),:])
    foldst.append(target[i:i+int(sampn/10),:])



#array to store data
MSE_test = np.zeros(10)
R2_test = np.zeros(10)
MSE_train = np.zeros(10)
R2_train = np.zeros(10)
k = np.zeros(10)
mse_keras_test = np.zeros(10)
r2_keras_test = np.zeros(10)
mse_keras_train = np.zeros(10)
r2_keras_train = np.zeros(10)


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
    train_targets = np.zeros((sumind,1))
    placedind = 0
    for j in range(0,10):
        if j != i and j != valind:
            train[placedind:placedind+foldsm[j].shape[0],:] = foldsm[j]
            train_targets[placedind:placedind+foldsm[j].shape[0],:] = foldst[j]
            placedind = placedind + foldsm[j].shape[0]




    # initialize the network
    mlp1 = mlp()


    #run the program
    k[i]=mlp1.earlystopping(train, train_targets, valid, valid_targets)


    #get the MSE and R2 for our model
    MSE_test[i], R2_test[i] = mlp1.score(test,test_targets)
    MSE_train[i], R2_train[i] = mlp1.score(train,train_targets)



    #using keras for comparison
    if mlp1.run_keras:
        model = Sequential()
        model.add(Dense(mlp1.hidden, input_dim=mlp1.ninput, activation='relu'))
        model.add(Dense(1, activation='linear'))
        sgd = optimizers.SGD(lr=mlp1.eta, momentum=mlp1.eta, decay=0.0, nesterov=False)
        model.compile(loss='mse', optimizer=sgd)
        earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')
        model.fit(train, train_targets, epochs=1000, verbose=0, batch_size=mlp1.minibatch, validation_data=(valid,valid_targets), callbacks=[earlystop])
        keras_pred_test = model.predict(test)
        keras_pred_train = model.predict(train)

        mse_keras_test[i] = met.mean_squared_error(test_targets,keras_pred_test)
        r2_keras_test[i] = met.r2_score(test_targets,keras_pred_test)
        mse_keras_train[i] = met.mean_squared_error(train_targets,keras_pred_train)
        r2_keras_train[i] = met.r2_score(train_targets,keras_pred_train)


    #print results for each fold
    print("Created MLP:")
    print("Test set MSE: %.4f R2: %.4f" %(MSE_test[i], R2_test[i]))
    print("Train set MSE: %.4f R2: %.4f\n" %(MSE_train[i], R2_train[i]))

    if mlp1.run_keras:
        print("keras MLP:")
        print("Test set MSE: %.4f R2: %.4f" %(mse_keras_test[i], r2_keras_test[i]))
        print("Train set MSE: %.4f R2: %.4f\n" %(mse_keras_train[i], r2_keras_train[i]))
    print("------------------\n")

#print results
print("Created MLP")
print("Test data:")
print("Average MSE %.2f. Min MSE: %.2f. Max MSE: %.2f. MSE std: %.2f"
%(np.mean(MSE_test), np.min(MSE_test), np.max(MSE_test), np.std(MSE_test)))
print("Average R2 %.2f. Min R2: %.2f. Max R2: %.2f. R2 std: %.2f"
%(np.mean(R2_test), np.min(R2_test), np.max(R2_test), np.std(R2_test)))
print("Train data:")
print("Average MSE %.2f. Min MSE: %.2f. Max MSE: %.2f. MSE std: %.2f"
%(np.mean(MSE_train), np.min(MSE_train), np.max(MSE_train), np.std(MSE_train)))
print("Average R2 %.2f. Min R2: %.2f. Max R2: %.2f. R2 std: %.2f\n"
%(np.mean(R2_train), np.min(R2_train), np.max(R2_train), np.std(R2_train)))

if mlp1.run_keras:
    print("keras MLP")
    print("Test data:")
    print("Average MSE %.2f. Min MSE: %.2f. Max MSE: %.2f. MSE std: %.2f"
    %(np.mean(mse_keras_test), np.min(mse_keras_test), np.max(mse_keras_test), np.std(mse_keras_test)))
    print("Average R2 %.2f. Min R2: %.2f. Max R2: %.2f. R2 std: %.2f"
    %(np.mean(r2_keras_test), np.min(r2_keras_test), np.max(r2_keras_test), np.std(r2_keras_test)))
    print("Train data:")
    print("Average MSE %.2f. Min MSE: %.2f. Max MSE: %.2f. MSE std: %.2f"
    %(np.mean(mse_keras_train), np.min(mse_keras_train), np.max(mse_keras_train), np.std(mse_keras_train)))
    print("Average R2 %.2f. Min R2: %.2f. Max R2: %.2f. R2 std: %.2f"
    %(np.mean(r2_keras_train), np.min(r2_keras_train), np.max(r2_keras_train), np.std(r2_keras_train)))

plt.figure(1)
# Plot  MSE on both the training and test data
plt.plot(MSE_train, 'b',label='Created MLP train')
plt.plot(MSE_test,'--b',label='Created MLP test')
if mlp1.run_keras:
    plt.plot(mse_keras_train,'r',label='keras MLP train')
    plt.plot(mse_keras_test,'--r',label='keras MLP test')



if mlp1.run_keras:
    plt.title("MSE-scores for test and training data for the created and the keras MLP during a 10-fold", fontsize = 16)
else:
    plt.title("MSE-scores for test and training data for the created MLP during a 10-fold", fontsize = 16)
plt.legend(fontsize=16)
plt.xlabel('Fold number',fontsize=15)
plt.ylabel('MSE - score',fontsize=15)
plt.tick_params(labelsize=15)


# Plot R2 on both the training and test data
plt.figure(2)
plt.plot(R2_train, 'b',label='Created MLP train')
plt.plot(R2_test,'--b',label='Created MLP test')
if mlp1.run_keras:
    plt.plot(r2_keras_train,'r',label='keras MLP train')
    plt.plot(r2_keras_test,'--r',label='keras MLP test')


if mlp1.run_keras:
    plt.title("R2-scores for test and training data for the created and the keras MLP during a 10-fold", fontsize = 16)
else:
    plt.title("R2-scores for test and training data for the created MLP during a 10-fold", fontsize = 16)
plt.legend(fontsize=16)
plt.xlabel('Fold number',fontsize=15)
plt.ylabel('R2 - score',fontsize=15)
plt.tick_params(labelsize=15)


plt.show()
