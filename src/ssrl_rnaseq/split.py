from itertools import chain, compress

import numpy as np
from sklearn.utils import check_random_state, shuffle
from sklearn.model_selection import StratifiedGroupKFold

__all__ = ["pretrain_downstream_split", "GroupKShotsFold"]


def pretrain_downstream_split(
        *arrays,
        pretrain_size,
        downstream_size,
        groups,
        stratify,
        random_state=None,
):
    """
    Split given `arrays` into pretrain and downstream sets.
    Uses StratifiedGroupKFold under the hood.
    """

    if pretrain_size < downstream_size:
        raise NotImplementedError

    indices = np.arange(len(stratify))

    n_splits = len(indices) // downstream_size

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pretrain, downstream = next(cv.split(indices, stratify, groups))

    pretrain = shuffle(pretrain, random_state=random_state)
    downstream = shuffle(downstream, random_state=random_state)

    pretrain = pretrain[: pretrain_size]
    downstream = downstream[: downstream_size]

    if len(pretrain) != pretrain_size or len(downstream) != downstream_size:
        raise ValueError

    return list(
        chain.from_iterable(
            (
                _index(a, pretrain),
                _index(a, downstream),
            )
            for a in arrays
        )
    )


class GroupKShotsFold:
    def __init__(self, n_splits, *, k, random_state=None):
        """
        Same as a StratifiedGroupKFold but only `k` samples per classes are kept in each training
        split.
        """

        self.n_splits = n_splits
        self.k = k
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups):
        y = np.asarray(y)

        random_state = check_random_state(self.random_state)

        classes, y = np.unique(y, return_inverse=True)
        classes = np.arange(len(classes))

        cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)

        for train, test in cv.split(X, y, groups):
            y_train = y[train]
            
            train = np.concatenate([
                random_state.choice(
                    train[y_train == class_],
                    size=self.k,
                    replace=False,
                )
                for class_ in classes
            ])
            
            train = shuffle(train, random_state=random_state)
            test = shuffle(test, random_state=random_state)

            yield train, test


def _index(X, indices):
    """
    From sklearn.
    """

    if hasattr(X, "iloc"):
        return X.iloc[indices]

    elif hasattr(X, "shape"):
        return X[indices]

    return [X[i] for i in indices]
