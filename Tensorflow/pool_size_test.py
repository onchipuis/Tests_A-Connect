import tensorflow as tf
import numpy as np
from Networks import MNIST_mismatch
from Scripts import load_ds
from Layers import AConnect

Wstd = [0.3,0.5,0.7]
Bstd = Wstd
Sim_err = [0,0.3,0.5,0.7]
isBin = ["no","yes"]
imgSize = [[28,28],[11,11]]
Q = [8,4]
mul = [1,2,4]
for p in range(len(mul)):
    for i in range(len(imgSize)):
        for j in range(len(Q)):
            for k in range(len(isBin)):
                for l in range(len(Wstd)):
                    wstd = str(int(100*Wstd[l]))
                    bstd = str(int(100*Bstd[l]))
                    name = 'A-Connect_'+wstd+'_'+bstd+'_'+'pool_'+str(mul[p])+'_batch' 
                    if(isBin[k]=="yes"):
                        name = namme+'_BW'
                    (x_train,y_train),(x_test,y_test) = load_ds.load_ds(imgSize=imgSize[i],Quant=Q[])                                                

