import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import linear_model
import random
import time as tm
from mpl_toolkits.axes_grid1 import ImageGrid


#perform the OLS regression
def OLS (X,Xte,zt,trainn):
    """
    Calculates the OLS regression using a polynomial.
    Inputs: Train data, test data, training outputs and number of training data.
    Outputs: Beta parameters, prediced z on train data and z predicted on test data.
    """
    #using SVD to find the pseudo inverse
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    beta = vh.T @ np.linalg.pinv(np.diag(s)) @ u.T @ zt
    zpred = X.dot(beta)
    zpredtest = Xte.dot(beta)
    return beta, zpred, zpredtest

#perform the ridge regression
def ridge (X,Xte,zt,lam,trainn):
    """
    Calculates the ridge regression using a polynomial.
    Inputs: Train data, test data, training outputs, lambda matrix  and number of training data.
    Outputs: Beta parameters, prediced z on train data and z predicted on test data.
    """
    beta = np.linalg.inv(X.T.dot(X) + lam).dot(X.T).dot(zt)
    zpred = X.dot(beta)
    zpredtest = Xte.dot(beta)
    return beta, zpred, zpredtest

#perform the lasso regression
def Lasso(X,Xte,zt,lamb,trainn):
    """
    Calculates the lasso regression using a polynomial.
    Inputs: Train data, test data, training outputs, lambda variable and number of training data.
    Outputs: Beta parameters, prediced z on train data and z predicted on test data.
    """
    lasso=linear_model.Lasso(alpha=lamb)
    lasso.fit(X,zt)
    zpred = lasso.predict(X)
    zpredtest = lasso.predict(Xte)
    beta = lasso.coef_
    return beta, zpred, zpredtest

#function to calculate MSE and R2
#needs flattend z and zpred array
def MSER2 (z,zpred,n):
    """
    Calculates the MSe and R2 scores.
    Inputs: Correct data, predicted data and number of data.
    Outputs: MSE- and R2-score.
    """
    ze = 0 #z error
    za = 0 #z sum of z - zavrage
    zm = np.mean(z) #calcute the mean of z
    #sum of error
    for i in range(0,n):
        ze = ze + (zpred[i] - z[i])**2
        za = za + (z[i] - zm)**2

    zMSE = ze/(n) #calcute MSE
    zR = 1 - (ze/za) #calcute R2

    return zMSE, zR,



#function to calculate Bias, variance and error terms of MSE
def VBE(zpred,z):
    """
    Calculates the bias and variance.
    Inputs: Correct data and predicted data.
    Outputs: bias and variance.
    """
    #gets the mean for the prediced values of z
    zpm = np.mean(zpred, axis=1, keepdims=True)

    #values to store sums
    zv = np.zeros(len(z))
    zb = 0
    n = len(z)

    for i in range(0,n):
        zb += (z[i] - zpm[i])**2

    for i in range(0,zpred.shape[1]):
            zv += zpred[:,i]**2

    #variance calculation
    varz = np.mean((zv / zpred.shape[1]) - zpm**2)

    #bias calculation
    biasz = (zb / n)


    # #used to compare methods
    # bias = np.mean( (z.reshape(len(z),1) - np.mean(zpred, axis=1, keepdims=True))**2 )
    # variance = np.mean( np.var(zpred, axis=1, keepdims=True) )
    # print(bias,variance)
    # print(biasz,varz)

    return varz, biasz

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

# adjustable parameters
sampn = 1000 #number of samples
lamb = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000] #value of lambda
trainp = 0.7 #Number of training samples given as %
bootrun = 100 #times to run the bootrstap
nlevel = 0 #noise level


#not adjustable
L = 40
count = 0 #counter for figures
#get number of training samples
trainn = int(sampn*trainp)
#function for noise
N0 = np.random.normal(0,nlevel, sampn)


# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(sampn,L))


#z calculated using energies
z = ising_energies(states,L) + N0


#get data for X matrix
states=np.einsum('...i,...j->...ij', states, states)
states=states.reshape((sampn,L*L))

#number of test data
testn = sampn - trainn

#array to store the r2 scores
train_r2_ols = np.zeros(len(lamb))
test_r2_ols = np.zeros(len(lamb))
train_r2_ridge = np.zeros(len(lamb))
test_r2_ridge = np.zeros(len(lamb))
train_r2_lasso = np.zeros(len(lamb))
test_r2_lasso = np.zeros(len(lamb))

#array to store the mse scores
train_mse_ols = np.zeros(len(lamb))
test_mse_ols = np.zeros(len(lamb))
train_mse_ridge = np.zeros(len(lamb))
test_mse_ridge = np.zeros(len(lamb))
train_mse_lasso = np.zeros(len(lamb))
test_mse_lasso = np.zeros(len(lamb))

#array to store bias and variance
bias_ols = np.zeros(len(lamb))
var_ols = np.zeros(len(lamb))
bias_ridge = np.zeros(len(lamb))
var_ridge = np.zeros(len(lamb))
bias_lasso = np.zeros(len(lamb))
var_lasso = np.zeros(len(lamb))

for l in lamb:

    #creates array to store data
    zeps = np.zeros((3,bootrun))
    zMSE = np.zeros((3,bootrun))
    zR = np.zeros((3,bootrun))
    zMSEte = np.zeros((3,bootrun))
    zRte = np.zeros((3,bootrun))
    timeend = np.zeros(bootrun)
    varz = np.zeros(3)
    biasz = np.zeros(3)

    #creates array to store the best beta coefficients
    beta = np.zeros((3,bootrun,L*L))

    #get random indices for test data
    randi = random.sample(range(0,sampn),testn)

    #array to store test data
    xte = np.zeros((testn,L*L))
    zte = np.zeros(testn)

    #get the test data from the full set
    for k in range(0,testn):
        xte[k] = states[randi[k],:]
        zte[k] = z[randi[k]]

    #remove the test data from the sample som they are not
    #used for training
    xtr = np.delete(states,randi,axis=0)
    ztr = np.delete(z,randi)

    #create array to store the test results
    zpredtest_o_sum = np.zeros((testn,bootrun))
    zpredtest_r_sum = np.zeros((testn,bootrun))
    zpredtest_l_sum = np.zeros((testn,bootrun))

    for j in range(0,bootrun):

        #set start time for the run
        timestart = tm.time()

        #get random indexes for the training data (with resampling)
        randi =np.random.randint((trainn), size=trainn)

        #array to store the trainging data
        xt = np.zeros((trainn,L*L))
        zt = np.zeros(trainn)

        #takes trainging data with resampling
        for k in range(0,trainn):
            xt[k] = xtr[randi[k],:]
            zt[k] = ztr[randi[k]]


        #fit functions
        #ridge
        lam = np.identity(xt.shape[1]) * l
        beta[1,j,:], zpred_r, zpredtest_r = ridge(xt,xte,zt,lam,sampn)

        #lasso
        beta[2,j,:], zpred_l, zpredtest_l = Lasso(xt,xte,zt,l,sampn)

        #OLS
        beta[0,j,:], zpred_o, zpredtest_o = OLS(xt,xte,zt,sampn)

        #calculate MSE and R2
        zMSE[0][j], zR[0][j] = MSER2(zt,zpred_o,len(zt))
        zMSE[1][j], zR[1][j] = MSER2(zt,zpred_r,len(zt))
        zMSE[2][j], zR[2][j] = MSER2(zt,zpred_l,len(zt))

        #calculate MSE and R2 for test data
        zMSEte[0][j], zRte[0][j] = MSER2(zte,zpredtest_o,len(zte))
        zMSEte[1][j], zRte[1][j] = MSER2(zte,zpredtest_r,len(zte))
        zMSEte[2][j], zRte[2][j] = MSER2(zte,zpredtest_l,len(zte))

        #sum the test results
        zpredtest_o_sum[:,j] = zpredtest_o
        zpredtest_r_sum[:,j] = zpredtest_r
        zpredtest_l_sum[:,j] = zpredtest_l

        #gets the times used to make and use model
        timeend[j] = tm.time() - timestart



    #function to calualte variance bias and error term
    varz[0], biasz[0] = VBE(zpredtest_o_sum,zte)
    varz[1], biasz[1] = VBE(zpredtest_r_sum,zte)
    varz[2], biasz[2] = VBE(zpredtest_l_sum,zte)


    #prints releevant data
    print("OLS method")
    print("Training data:")
    print("MSE = %.3f   R2 = %.3f" %(np.mean(zMSE[0,:],) , np.mean(zR[0,:])))
    print("Test data:")
    print("MSE = %.3f   R2 = %.3f" %(np.mean(zMSEte[0,:]) , np.mean(zRte[0,:])))
    print("Bias     : %.4f" %(np.mean(biasz[0])))
    print("Variance : %.4f\n" %(np.mean(varz[0])))


    print("Ridge method with lambda=%.4f" %l)
    print("Training data:")
    print("MSE = %.3f   R2 = %.3f" %(np.mean(zMSE[1,:],) , np.mean(zR[1,:])))
    print("Test data:")
    print("MSE = %.3f   R2 = %.3f" %(np.mean(zMSEte[1,:]) , np.mean(zRte[1,:])))
    print("Bias     : %.4f" %(np.mean(biasz[1])))
    print("Variance : %.4f\n" %(np.mean(varz[1])))

    print("Lasso method with lambda=%.4f" %l)
    print("Training data:")
    print("MSE = %.3f   R2 = %.3f" %(np.mean(zMSE[2,:],) , np.mean(zR[2,:])))
    print("Test data:")
    print("MSE = %.3f   R2 = %.3f" %(np.mean(zMSEte[2,:]) , np.mean(zRte[2,:])))
    print("Bias     : %.4f" %(np.mean(biasz[2])))
    print("Variance : %.4f\n" %(np.mean(varz[2])))

    print("-------------------------------------------------")

    #store data for plotting
    train_r2_ols[count] = np.mean(zR[0,:])
    test_r2_ols[count] = np.mean(zRte[0,:])
    train_r2_ridge[count] = np.mean(zR[1,:])
    test_r2_ridge[count] = np.mean(zRte[1,:])
    train_r2_lasso[count]= np.mean(zR[2,:])
    test_r2_lasso[count] = np.mean(zRte[2,:])

    train_mse_ols[count] = np.mean(zMSE[0,:])
    test_mse_ols[count] = np.mean(zMSEte[0,:])
    train_mse_ridge[count] = np.mean(zMSE[1,:])
    test_mse_ridge[count] = np.mean(zMSEte[1,:])
    train_mse_lasso[count]= np.mean(zMSE[2,:])
    test_mse_lasso[count] = np.mean(zMSEte[2,:])

    bias_ols[count] = np.mean(biasz[0])
    var_ols[count] = np.mean(varz[0])
    bias_ridge[count] = np.mean(biasz[1])
    var_ridge[count] = np.mean(varz[1])
    bias_lasso[count] = np.mean(biasz[2])
    var_lasso[count] = np.mean(varz[2])


    #method for plotting gotten from stack exchange. Vissited: 02-11-18
    #url: https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    #author: spinup. url: https://stackoverflow.com/users/1329892/spinup
    fig = plt.figure(figsize=(9.75, 3))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1,3),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.15,
                     )

    # Add data to image grid
    grid[0].imshow(beta[0][j,:].reshape(L,L), cmap='seismic',vmin=-1, vmax=1,)
    grid[0].set_title('OLS',fontsize=16)

    grid[1].imshow(beta[1][j,:].reshape(L,L), cmap='seismic',vmin=-1, vmax=1,)
    grid[1].set_title('Ridge $\\lambda=%.4f$'%l,fontsize=16)

    im = grid[2].imshow(beta[2][j,:].reshape(L,L), cmap='seismic',vmin=-1, vmax=1,)
    grid[2].set_title('Lasso $\\lambda=%.4f$'%l,fontsize=16)
    # Colorbar
    grid[2].cax.colorbar(im)
    grid[2].cax.toggle_label(True)

    count += 1

plt.figure(count +1)
# Plot our performance on both the training and test data
plt.semilogx(lamb, train_r2_ols, 'b',label='OLS train')
plt.semilogx(lamb, test_r2_ols,'--b',label='OLS test')
plt.semilogx(lamb, train_r2_ridge,'r',label='Ridge train',linewidth=1)
plt.semilogx(lamb, test_r2_ridge,'--r',label='Ridge test',linewidth=1)
plt.semilogx(lamb, train_r2_lasso, 'g',label='Lasso train')
plt.semilogx(lamb, test_r2_lasso, '--g',label='Lasso test')


#fig.set_size_inches(10.0, 6.0)
plt.title("R2-scores for test and training data with different values for lambda", fontsize = 16)
plt.legend(loc='lower left',fontsize=16)
plt.ylim([-0.01, 1.01])
plt.xlim([min(lamb), max(lamb)])
plt.xlabel('Lambda',fontsize=15)
plt.ylabel('R2 - score',fontsize=15)
plt.tick_params(labelsize=15)

plt.figure(count + 2)
# Plot our performance on both the training and test data
plt.semilogx(lamb, train_mse_ols, 'b',label='OLS train')
plt.semilogx(lamb, test_mse_ols,'--b',label='OLS test')
plt.semilogx(lamb, train_mse_ridge,'r',label='Ridge train',linewidth=1)
plt.semilogx(lamb, test_mse_ridge,'--r',label='Ridge test',linewidth=1)
plt.semilogx(lamb, train_mse_lasso, 'g',label='Lasso train')
plt.semilogx(lamb, test_mse_lasso, '--g',label='Lasso test')


#fig.set_size_inches(10.0, 6.0)
plt.title("MSE-scores for test and training data with different values for lambda", fontsize = 16)
plt.legend()
#plt.ylim([-0.01, 1.01])
plt.xlim([min(lamb), max(lamb)])
plt.xlabel('Lambda',fontsize=15)
plt.ylabel('MSE - score',fontsize=15)
plt.tick_params(labelsize=15)

plt.figure(count + 3)
# Plot our performance on both the training and test data
plt.semilogx(lamb, bias_ols, 'b',label='OLS bias')
plt.semilogx(lamb, var_ols,'--b',label='OLS var')
plt.semilogx(lamb, bias_ridge,'r',label='Ridge bias',linewidth=1)
plt.semilogx(lamb, var_ridge,'--r',label='Ridge var',linewidth=1)
plt.semilogx(lamb, bias_lasso, 'g',label='Lasso bias')
plt.semilogx(lamb, var_lasso, '--g',label='Lasso var')


#fig.set_size_inches(10.0, 6.0)
plt.title("Bias and variance for test data with different values for lambda", fontsize = 16)
plt.legend()
#plt.ylim([-0.01, 1.01])
plt.xlim([min(lamb), max(lamb)])
plt.xlabel('Lambda',fontsize=15)
plt.ylabel('bias/var - score',fontsize=15)
plt.tick_params(labelsize=15)



plt.show()
