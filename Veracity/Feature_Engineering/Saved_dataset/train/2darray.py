import numpy as np

x_train=np.load('train_array.npy')

print (x_train.shape)

print (x_train[100].shape)
print (x_train[0][1])

X_train=x_train[0]

#for i in range(1,x_train.shape[0]):
  #X_train=np.vstack((X_train,x_train[i]))

print (X_train.shape)
