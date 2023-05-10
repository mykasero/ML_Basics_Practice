import numpy as np

rand_a = np.random.rand()
rand_a

rand_a = np.random.rand(2,5)
rand_a

rand_b = np.random.rand(1,5)

rand_b = rand_b *100

rand_c = np.vstack((rand_a,rand_b))

rand_b = np.random.rand(2,1)

rand_c = np.hstack((rand_a,rand_b))

X = rand_c[:,1]

a,b = 3,8

y = a*X + b

X = np.random.randint(0,10,10)

X = X*10

# print(np.random.choice(10,size = 3, replace=False))

idx = np.random.choice(X.shape[0],size = 3, replace=False)

X[idx]
