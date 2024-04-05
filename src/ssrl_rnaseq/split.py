from itertools import chain

import numpy as np
from sklearn.utils import check_random_state, shuffle
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

__all__ = ["pretrain_downstream_split", "GroupKShotsFold"]


def pretrain_downstream_split(
        *arrays,
        pretrain_size=None,
        downstream_size=None,
        groups,
        stratify=None,
        random_state=None,
):
    """
    Split given `arrays` into pretrain and downstream sets.
    The pretrain set contains `pretrain_size` groups.
    The pretrain set contains `downstream_size` groups.
    Notice that the sizes are not numbers of samples but number of groups.
    """

    unique_groups, groups = np.unique(groups, return_inverse=True)
    unique_groups = len(unique_groups)

    # Attribute the group's most common class to each group
    if stratify is not None:
        stratify = np.asarray(stratify)

        unique_classes, stratify = np.unique(stratify, return_inverse=True)
        unique_classes = len(unique_classes)

        s = np.zeros((unique_groups, unique_classes), dtype=np.int64)
        np.add.at(s, (groups, stratify), 1)

        stratify = s.argmax(axis=1)

    # Split
    pretrain_groups, downstream_groups = train_test_split(
        np.arange(unique_groups),
        train_size=pretrain_size,
        test_size=downstream_size,
        shuffle=True,
        stratify=stratify,
        random_state=random_state,
    )

    pretrain = np.isin(groups, pretrain_groups)
    downstream = np.isin(groups, downstream_groups)

    return list(
        chain.from_iterable(
            (a[pretrain], a[downstream]) for a in arrays
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
