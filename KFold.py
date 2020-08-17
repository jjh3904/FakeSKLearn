import numpy as np
import random
import pandas as pd

class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits=n_splits
        self.random_state=random_state
        self.shuffle=shuffle

    def split(self, x, y=None):
        x=np.array(x)

        indices=np.arange(x.shape[0])

        if self.shuffle:
            for l in range(x.shape[0]-1,-1,-1):
                np.random.RandomState(self.random_state).shuffle(indices)

        n_splits=self.n_splits

        fold_sizes=np.full(n_splits, x.shape[0]//n_splits, dtype=np.int)
        fold_sizes[:x.shape[0] % n_splits]+=1

        current=0

        for fold_size in fold_sizes:
            test_mask=np.zeros(x.shape[0], dtype=np.bool)
            test_mask[current : current + fold_size]=True
            yield (
                    indices[np.logical_not(test_mask)],
                    indices[test_mask]
                    )
            current+=fold_size

    def get_n_splits(self):
        return self.n_splits

if __name__=='__main__':
    kf=KFold()
    train=pd.read_csv('train.csv')
    test=pd.read_csv('y_train.csv')

    for xtrain, xtest in kf.split(train,test):
        print (train.values[xtrain])
        print ('\n')
        print (train.values[xtest])
