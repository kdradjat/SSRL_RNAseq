import numpy as np
from sklearn.utils import check_random_state, shuffle
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

__all__ = ["GroupKShotsFold"]


def pretext_downstream_split(
        *array,
        pretext_size=None,
        downstream_size=None,
        groups,
        stratify=None,
        random_state=None,
):
    unique_groups = np.unique(groups)

    pretext_groups, downstream_groups = train_test_split(
        unique_groups,
        train_size=pretext_size,
        downstream_size=downstream_size,
        shuffle=True,
        stratify=stratify,
        random_state=random_state,
    )

    pretext = groups.isin(pretext_groups)
    downstream = groups.isin(downstream_groups)

    return pretext


class GroupKShotsFold:
    def __init__(self, n_splits, *, k, random_state=None):
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
