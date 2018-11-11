import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import linear_model
import random
import time as tm
import pickle


def predict(x,beta):
    """
    Function that predicts labels based on the beta parameters that has been
    calculatted.
    Inputs: datapoints and beta parameters
    Output: predicted label
    """
    ypred = x @ beta
    ypred = 1 / (1 + np.exp(-ypred))
    ypred[ypred < 0.5] = 0
    ypred[ypred >= 0.5] = 1
    return ypred


def beta_update(lr,x,y,beta):
    """
    Function to calculate the new beta parameters without regularization.
    Inputs: data, labels, beta parameters
    Output: updated beta
    """
    output = 1 / (1 + np.exp(-(x @ beta)))
    delta = x.T @ (output - y.reshape((len(y),1)))
    new_beta = beta - lr * delta/len(y)
    return new_beta

def beta_update_l2(lr,x,y,beta,lamb):
    """
    Function to calculate the new beta parameters with l1 regularization.
    Inputs: data, labels, beta parameters, lambda
    Output: updated beta
    """
    output = 1 / (1 + np.exp(-(x @ beta)))
    delta = x.T @ (output - y.reshape((len(y),1)))
    new_beta = beta - lr * (delta/len(y) + lamb * beta)
    return new_beta

#adjustable parameters
trainp = 0.5 #percentage of the data to be used for training
minibatch = 20 #minibatch size
lr = 0.00001 #learning rate
lamb = [0.00001, 0.0001, 0.01, 1, 100, 10000, 100000] #value of lambda

#not adjustable
L = 40 #the root of the number of spins


#set the filenames
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


# Set up slices of the dataset
ordered = slice(0, 70000)
critical = slice(70000, 100000)
disordered = slice(100000, 160000)

#creates a dataset without the critical data
datawo = np.concatenate((data[ordered], data[disordered]))
labelswo = np.concatenate((labels[ordered], labels[disordered]))

#creates array to store the critical data
critical_data = np.zeros((30000,L*L+1))
critical_data[:,0] = -1
critical_data[:,1:] = data[critical]
critical_label = labels[critical]

#randmoly shuffle the data
order = list(range(np.shape(datawo)[0]))
np.random.shuffle(order)
datawo = datawo[order,:]
labelswo = labelswo[order]

#find total samples and calculate the number of training data
sampn = len(labelswo)
trainn = int(sampn*trainp)


#split the data into training and test sets
xt = np.zeros((trainn,L*L+1))
xt[:,0] = -1
xt[:,1:] = datawo[:trainn,:]
yt = labelswo[:trainn]

xte = np.zeros((sampn-trainn,L*L+1))
xte[:,0] = -1
xte[:,1:] = datawo[trainn:,:]
yte = labelswo[trainn:]


#array to store the ac scores
train_ac = np.zeros(len(lamb))
test_ac = np.zeros(len(lamb))
citical_ac = np.zeros(len(lamb))
train_ac_l2 = np.zeros(len(lamb))
test_ac_l2 = np.zeros(len(lamb))
citical_ac_l2 = np.zeros(len(lamb))
train_ac_sci = np.zeros(len(lamb))
test_ac_sci = np.zeros(len(lamb))
citical_ac_sci = np.zeros(len(lamb))

count = 0
for l in lamb:

    #initialize the beta parameters
    beta = (2/np.sqrt(L*L+1)) * np.random.random_sample((L*L+1,1)) -1/np.sqrt(L*L+1)
    beta_l2 = (2/np.sqrt(L*L+1)) * np.random.random_sample((L*L+1,1)) -1/np.sqrt(L*L+1)

    #variable to store the best score
    best_score = 0
    best_score_l2 = 0
    for k in range(0,50):

        for i in range(0,trainn,minibatch):

            #update beta parameters
            beta = beta_update(lr,xt[i:i+minibatch,:],yt[i:i+minibatch],beta)
            beta_l2 = beta_update_l2(lr,xt[i:i+minibatch,:],yt[i:i+minibatch],beta_l2,l)


        #checks the score of the model and stores the best parameters
        temp_pred = predict(xt,beta)
        temp_score = np.sum(yt.reshape((trainn,1)) == temp_pred) / len(yt)
        if temp_score > best_score:
            best_beta = beta
            best_score = temp_score

        temp_pred = predict(xt,beta_l2)
        temp_score = np.sum(yt.reshape((trainn,1)) == temp_pred) / len(yt)
        if temp_score > best_score_l2:
            best_beta_l2 = beta_l2
            best_score_l2 = temp_score


        #reshuffle the data so the model does not train the same way
        order = list(range(np.shape(xt)[0]))
        np.random.shuffle(order)
        xt = xt[order,:]
        yt = yt[order]

    #predicts the labels using beta
    ypred = predict(xte,best_beta)
    ypred_train = predict(xt,best_beta)
    critical_ypred = predict(critical_data,best_beta)

    #predicts the labels using beta_l2
    ypred_l2 = predict(xte,best_beta_l2)
    ypred_train_l2 = predict(xt,best_beta_l2)
    critical_ypred_l2 = predict(critical_data,best_beta_l2)

    #fiting a scikit model
    scilearn = linear_model.LogisticRegression(penalty='l2',C=1/l).fit(xt, yt)


    #calualte the score of the predicted labels
    test_ac[count]= np.sum(yte.reshape((sampn - trainn,1)) == ypred) / len(yte)
    train_ac[count]= np.sum(yt.reshape((trainn,1)) == ypred_train) / len(yt)
    citical_ac[count] = np.sum(critical_label.reshape((len(critical_label),1)) == critical_ypred) / len(critical_label)

    test_ac_l2[count]= np.sum(yte.reshape((sampn - trainn,1)) == ypred_l2) / len(yte)
    train_ac_l2[count] = np.sum(yt.reshape((trainn,1)) == ypred_train_l2) / len(yt)
    citical_ac_l2[count] = np.sum(critical_label.reshape((len(critical_label),1)) == critical_ypred_l2) / len(critical_label)

    #calualte the score of the scikit models
    test_ac_sci[count] = scilearn.score(xte,yte)
    train_ac_sci[count] = scilearn.score(xt,yt)
    citical_ac_sci[count] = scilearn.score(critical_data, critical_label)




    #print results
    print("Created minibatch method:")
    print("Train score: %.4f" %train_ac[count])
    print("Test score: %.4f" %test_ac[count])
    print("Critical score: %.4f\n" %citical_ac[count])

    print("Created minibatch with L2 regularization lambda = %.5f:" %l)
    print("Train score: %.4f" %train_ac_l2[count])
    print("Test score: %.4f" %test_ac_l2[count])
    print("Critical score: %.4f\n" %citical_ac_l2[count])

    print("Scikit learn method:")
    print("Train score: %.4f" %train_ac_sci[count])
    print("Test score: %.4f" %test_ac_sci[count])
    print("Critical score: %.4f\n" %citical_ac_sci[count])

    print("--------------\n")

    count += 1
plt.figure(count +1)
# Plot our performance on both the training and test data
plt.semilogx(lamb, train_ac, 'b',label='Created method train')
plt.semilogx(lamb, test_ac,'--b',label='Created method test')
plt.semilogx(lamb, citical_ac,'-.b',label='Created method critical')
plt.semilogx(lamb, train_ac_l2,'r',label='Created L2 train',linewidth=1)
plt.semilogx(lamb, test_ac_l2,'--r',label='Created L2 test',linewidth=1)
plt.semilogx(lamb, citical_ac_l2,'-.r',label='L2 critical',linewidth=1)
plt.semilogx(lamb, train_ac_sci, 'g',label='Scikit train')
plt.semilogx(lamb, test_ac_sci, '--g',label='Scikit test')
plt.semilogx(lamb, citical_ac_sci, '-.g',label='Scikit critical')


plt.title("Accuracy scores for test and training data with different values for lambda", fontsize = 16)
plt.legend(loc='lower left',fontsize=16)
plt.ylim([0.4, 0.8])
plt.xlim([min(lamb), max(lamb)])
plt.xlabel('Lambda',fontsize=15)
plt.ylabel('R2 - score',fontsize=15)
plt.tick_params(labelsize=15)


plt.show()
