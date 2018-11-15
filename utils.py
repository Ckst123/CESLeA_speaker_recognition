#this file is used for changing the transition matrix for state transitions
#we can use transitions of our choice by calling the hmm.transmat_prior function
import numpy as np
tr = []
#16 number of states
for i in range (16):
    #print(i)
    q = []
    for j in range(16):

        #print(j)
        if j== 0:
            a = 0.6
        elif j== 1:
            a = 0.4
        else:
            a = 0
        if i ==15 and j == 1:
            print(i)
            a = 0
        q.append(a)
    #q = np.array(q)
    q = np.roll(q,i)
    #q = q.astype(list)
    tr.append(q)
tr = np.array(tr)
#print(tr)

