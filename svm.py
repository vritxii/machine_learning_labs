import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_circles
from ipywidgets import interact, fixed
from sklearn.svm import SVC 
from mpl_toolkits import mplot3d

def gen_dataset(x_range=5, yrange=5, sample_number=100):
    Sigma = np.array([[1, 0], [0, 1]])
    R = cholesky(Sigma)

    mu1 = np.array([[x_range, yrange]])
    x1 = np.dot(np.random.randn(sample_number, 2), R) + mu1
    y1=np.zeros(np.shape(x1)[1])
    x1=x1.T 

    mu2 = np.array([[-x_range, yrange]])
    x2= np.dot(np.random.randn(sample_number, 2), R) + mu2
    y2=np.ones(np.shape(x2)[1])
    x2=x2.T

    mu3 = np.array([[-x_range, -yrange]])
    x3= np.dot(np.random.randn(sample_number, 2), R) + mu3
    y3=np.zeros(np.shape(x3)[1])
    x3=x3.T

    mu4 = np.array([[x_range, -yrange]])
    x4= np.dot(np.random.randn(sample_number, 2), R) + mu4
    y4=np.ones(np.shape(x4)[1])
    x4=x4.T
    return x1,y1,x2,y2,x3,y3,x4,y4

def stablexy(x,y, x_range = 10, y_range = 10):
    N = len(x);
    x_min = 0
    x_max = N
    y_min = 0
    y_max = N
    for i in range(N):
        if x[i] < (-1.2*x_range):
            x_min += 1
        elif x[i] > 1.2*x_range:
            x_max -= 1
            pass

        if y[i] < (-1.2*y_range):
            y_min += 1
        elif y[i] > 1.2*y_range:
            y_max -= 1
            pass
    xm = y_min
    if x_min > y_min:
        xm = x_min
    ym = y_max
    if x_max > y_max:
        ym = x_max
    return x[xm:ym], y[xm:ym]
    pass

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    pass

def ex1():
    x1,y1,x2,y2,x3,y3,x4,y4 = gen_dataset()

    plt.plot(x1[0,:],x1[1,:],'go')
    plt.plot(x2[0,:],x2[1,:],'yo')
    plt.plot(x3[0,:],x3[1,:],'bo')
    plt.plot(x4[0,:],x4[1,:],'ro')
    plt.show()

    #build array
    x0=np.array([np.concatenate((x1[0],x3[0])),np.concatenate((x1[1],x3[1]))])
    x1=np.array([np.concatenate((x2[0],x4[0])),np.concatenate((x2[1],x4[1]))])

    y=np.array([np.concatenate((y1,y2,y3,y4))])

    x=np.array([np.concatenate((x0[0],x1[0])),np.concatenate((x0[1],x1[1]))])

    m=np.shape(x)[1]
    print('m = ', m)
    phi=(1.0/m)*len(y1)
    u0=np.mean(x0,axis=1)  
    u1=np.mean(x1,axis=1)

    xplot0=x0;xplot1=x1   #save the original data  to plot 
    x0=x0.T;x1=x1.T;x=x.T
    x0_sub_u0=x0-u0
    x1_sub_u1=x1-u1
    x_sub_u=np.concatenate([x0_sub_u0,x1_sub_u1])

    x_sub_u=np.mat(x_sub_u)
    sigma=(1.0/m)*(x_sub_u.T*x_sub_u)

    #plot the  discriminate boundary ,use the u0_u1's midnormal
    midPoint=[(u0[0]+u1[0])/2.0,(u0[1]+u1[1])/2.0]
    k=(u1[1]-u0[1])/(u1[0]-u0[0])
    x=range(-2,11)
    y=[(-1.0/k)*(i-midPoint[0])+midPoint[1] for i in x]

    #plot the figure and add comments
    plt.figure(1)
    plt.clf()
    plt.plot(xplot0[0],xplot0[1],'go')
    plt.plot(xplot1[0],xplot1[1],'yo')
    print(x,y)
    x, y = stablexy(x, y, 2, 11)
    plt.plot(x,y)
    plt.title("Gaussian discriminat analysis")
    plt.show()
    pass

def ex2():
    X, y = make_circles(100, factor=.1, noise=.1)
    clf = SVC(kernel='linear').fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
    plt.show()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
    plot_svc_decision_function(clf);
    plt.show()

    r = np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='summer')
    ax.view_init(elev=30, azim=30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()
    clf = SVC(kernel='rbf')
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
    plot_svc_decision_function(clf)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=200, facecolors='none');
    plt.show()
    pass

if __name__ == '__main__':
    #ex1()
    ex2()