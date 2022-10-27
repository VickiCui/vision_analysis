import numpy as np

def average_precision(gt, pred):
    """
    Computes the average precision.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    gt: set
        A set of ground-truth elements (order doesn't matter)
    pred: list
            A list of predicted elements (order does matter)

    Returns
    -------
    score: double
        The average precision over the input lists
    """

    if not gt:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(pred):
        if p in gt and p not in pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / max(1.0, len(gt))

def metrics(gt, pred, metrics_map):
    '''
    Returns a numpy array containing metrics specified by metrics_map.
    gt: set
        A set of ground-truth elements (order doesn't matter)
    pred: list
        A list of predicted elements (order does matter)
    '''
    if "@" in metrics_map:
        k = int(metrics_map.split("@")[-1])
        pred = pred[:k]

    if "MAP" in metrics_map:
        avg_precision = average_precision(gt=gt, pred=pred)
        return avg_precision

    if "HIT" in metrics_map:
        count = 0
        for item in pred:
            if item in gt:
                count += 1
        return count

    if ('RPrec' in metrics_map):
        intersec = len(gt & set(pred[:len(gt)]))
        return intersec / max(1., float(len(gt)))

    if "MRR" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred):
            if item in gt:
                score = 1.0 / (rank + 1.0)
                break
        return score


