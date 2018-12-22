#!/usr/bin/env python
# coding: utf-8

# In[1]:


####import libraries

#set GPU to use
import os
os.environ["THEANO_FLAGS"] = "device=cuda1,floatX=float32,exception_verbosity='high'"

#theano
from theano import *
import theano.tensor as T


# ### Class for Deep Embedding Kernel
# 
# Classification deep embedding kernel for tabular data

# In[2]:


class DEK(object):
    
    # rng: random generator
    # input: theano variable for input data
    # n_in: input size, integer
    # l_layers: architecture of lower (embedding) layers, must be a list
    # h_layers: architecture of higher (kernel) layers, must be a list
    # n_out: output size, integer
    # activation: activation function, must be theano function
    def __init__(self,rng,input,n_in,l_layers,h_layers,n_out,activation=T.nnet.relu):
        
        self.nl = len(l_layers)
        self.nl_out = l_layers[-1]
        self.nh = len(h_layers)
        self.input = input
        
        #function to initialize model weights
        def genW(size):
            return np.asarray(
                rng.uniform(
                    low=-0.1,
                    high=0.1,
                    size = size
                ),
                dtype = theano.config.floatX)
        
        self.params = []
        
        #computational flow
        
        #embedding layer initialization
        self.LLayer_W = []
        self.LLayer_b = []
        for i in range(self.nl):
            if i==0:
                size_in = n_in
            else:
                size_in = l_layers[i-1]
            self.LLayer_W.append(theano.shared(genW([size_in,l_layers[i]]),name='W'+str(i),borrow=True))
            self.LLayer_b.append(theano.shared(np.zeros(l_layers[i],dtype = theano.config.floatX),name='b'+str(i),borrow=True))
            self.params += [self.LLayer_W[i],self.LLayer_b[i]]
        
        #embeddings of stream 1
        u_x = self.input[0]
        for i in range(self.nl):
            u_x = activation(T.dot(u_x,self.LLayer_W[i]) + self.LLayer_b[i])
        #embedding of stream 2
        u_y = self.input[1]
        for i in range(self.nl):
            u_y = activation(T.dot(u_y,self.LLayer_W[i]) + self.LLayer_b[i])
            
        #combine embedding
        self.u1 = T.abs_(u_x - u_y)
        self.u2 = u_x * u_y
        self.MW1 = theano.shared(genW([l_layers[-1],h_layers[0]]),name='WM1',borrow=True)
        self.MW2 = theano.shared(genW([l_layers[-1],h_layers[0]]),name='WM2',borrow=True)
        self.Mb = theano.shared(np.zeros(h_layers[0],dtype = theano.config.floatX),name='bM',borrow=True)
        self.u = activation(T.dot(self.u1,self.MW1)+T.dot(self.u2,self.MW2)+self.Mb)
        self.params += [self.MW1,self.MW2,self.Mb]
        
        #kernel layer initialization
        self.HLayer_W = []
        self.HLayer_b = []
        for i in range(self.nh):
            if i==0:
                size_in = h_layers[0]
            else:
                size_in = h_layers[i-1]
            self.HLayer_W.append(theano.shared(genW([size_in,h_layers[i]]),name='W'+str(i+self.nl),borrow=True))
            self.HLayer_b.append(theano.shared(np.zeros(h_layers[i],dtype = theano.config.floatX),name='b'+str(i+self.nl),borrow=True))
            self.params += [self.HLayer_W[i],self.HLayer_b[i]]
        
        #output layer
        self.outLayer_W = theano.shared(genW([h_layers[-1],n_out]),name='Wout',borrow=True)
        self.outLayer_b = theano.shared(np.ones(n_out,dtype = theano.config.floatX),name='bout',borrow=True)
        self.params += [self.outLayer_W,self.outLayer_b]
        
        #kernel stream 
        K = self.u
        for i in range(self.nh):
            K = activation(T.dot(K,self.HLayer_W[i]) + self.HLayer_b[i])
        
        self.K = T.nnet.relu(T.dot(K,self.outLayer_W)+self.outLayer_b)
    
    #loss function 1
    def MSE(self,y):
        return T.mean((self.K - y)**2)
    
    #loss function 2
    def CE(self,y):
        return T.nnet.binary_crossentropy(self.K,y).mean()
    
    #predict kernel for a single data stream
    def predict(self,X):
        sX = theano.shared(X,borrow=True)
        fit = theano.function([], self.K, givens={self.input : sX})
        sX = [] #clear memory
        return fit()
    
    #predict kernel for two data stream
    def kernel(self,X,Y):
        i1 = np.repeat(np.arange(X.shape[0]),repeats=Y.shape[0])
        i2 = np.tile(np.arange(Y.shape[0]),reps=X.shape[0])
        sX = theano.shared(np.stack([X[i1].astype(np.float32),Y[i2].astype(np.float32)]),borrow=True)
        fit = theano.function([], self.K, givens={self.input : sX})
        sX = []
        return fit().reshape(X.shape[0],Y.shape[0])


# ### function to create and initialize a deep embedding kernel

# In[3]:


#dX: feature data
#hiddens: deep embedding kernel architecture, 
#   consists of a list for embedding network 
#   and a list for kernel network
def create_DEK(dX,
              hiddens=[[50,50],[50,50]],
              seed=12345):
    
    np_rng = np.random.RandomState(seed)
    
    x = T.tensor3('x')
    
    dk = DK(rng=np_rng,
            input=x,
            n_in=dX.shape[2],
            l_layers=hiddens[0],
            h_layers=hiddens[1],
            n_out=1,
           )
    
    return dk


# ### function to train an initialized deep embedding kernel

# In[ ]:


#dek: initialized deep embedding kernel
#dX: feature data
#dY: label data
def train_DEK(dek,
              dX,
              dY,
              learning_rate=0.01,
              n_epochs=1000,
              batch_size=20,
              seed=12345):
    
    trainX = theano.shared(np.asarray(dX, dtype=theano.config.floatX))
    trainY = theano.shared(np.asarray(dY, dtype=theano.config.floatX))        
    n_train_batches = dX.shape[1] // batch_size    
    y = T.matrix('y')
    cost = dk.CE(y)
    index = T.iscalar()        
    gparams = [T.grad(cost,param) for param in dk.params]
    updates = [(param, param - learning_rate * gparam) for param,gparam in zip(dk.params,gparams)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            dk.input: trainX[:,index*batch_size : (index+1)*batch_size],
            y: trainY[index*batch_size : (index+1)*batch_size]
        }
    )
    
    ####
    #training
    print "...training"    
    epoch = 0
    while (epoch < n_epochs):
        epoch = epoch + 1       
        for b_index in range(n_train_batches):
            c = []
            c.append(train_model(b_index))
            iter = (epoch-1) * n_train_batches + b_index            
            if (iter + 1) % n_train_batches == 0:
                print(
                    'epoch %i, minibatch %i/%i, current cost %f' %
                    (
                        epoch,
                        b_index + 1,
                        n_train_batches,
                        np.mean(c)
                    )
                )
                print c
    print("Optimization completed")
         
    return dek


# ### utility functions

# In[4]:


#function to filter top rc% nearest neighbors for each instance
#then generate paired data with similarity
#X: feature data
#Y: label data
#rc: top % nearest neighbors
#kernel: kernel/similarity function
def rank_filter(X, Y, rc, kernel):
    n = X.shape[0]
    d = kernel(X,X)
    s_ind = np.argsort(d,axis=1)
    kY = ((Y==Y.T)*1)
    sorted_kY = kY[np.repeat(np.arange(n),repeats=n),s_ind.flatten()].reshape(n,n)
    csum_kY = np.cumsum(sorted_kY,axis=1)
    rec_kY = csum_kY / np.repeat((np.sum(kY,axis=1)+0.0),repeats=n).reshape(n,n)
    fx = s_ind[rec_kY <= rc]
    fy = np.repeat(np.arange(n),repeats=n).reshape(n,n)[rec_kY <= rc]
    return fx, fy

#function to generate a gram matrix
#oD: original dimension
#data: similarity data
def gen_gram(oD, data):
    gram = np.zeros((oD,oD), dtype=np.float32)
    gram[np.triu_indices(oD)] = data
    gram = gram + gram.T
    gram[np.diag_indices(oD)] = 1.
    return gram