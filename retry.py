import numpy as np

class KFold:
    def __init__(self, n_splits=5, random_state=42, shuffle=False):
        self.n_splits=n_splits
        self.random_state=random_state
        self.shuffle=shuffle

    def split(self, x, y=None):

        x=np.array(x)
        indices=np.arange(x)

        if self.shuffle:
            np.random.seed(self.random_state).shuffle(indices)

        n_splits=self.n_splits
        fold_sizes=np.full(n_splits, x.shape[0]//n_splits, dtype=np.int)
        fold_sizes[:x.shape[0] % n_splits]+=1

        current=0
        for fold in fold_sizes:
            end=current+fold
            test_mask=np.zeros(x.shape[0], dtype=np.bool)
            tesk_mask[current:end]=True

            yield(
                    indices[np.logical_not(test_mask)],
                    indices[test_mask]
                    )
            current=end

class GroupKFold:
    def __init__(self, n_splits=5):
        self.n_splits=5

    def split(self, x, y, groups):
        x=np.array(x)
        unique_groups, groups=np.unique(groups, return_inverse=True)

        num_groups=len(unique_groups)

        n_samples_per_group=np.bincount(groups)
        n_samples_per_fold=np.zeros(n_splits)

        indices=np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group=n_samples_per_group[indices]

        group_to_fold=np.zeros(n_splits)

        for group_idx, weight in enumerate(n_samples_per_group):
            lightest_fold=np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_idx]] = lightest_fold

        yield_indices=np.arange(x.shape[0])

        for fold in range(self.n_splits):
            test_mask = np.zeros(x[0].shape, dtype=np.bool)
            test_mask[np.where(indices == f)[0]]=True

            yield(
                    yield_indices[np.logical_not(test_mask)],
                    yield_indices[test_mask]
                    )

