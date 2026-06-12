def intra_list_diversity(items):

    if len(items) <= 1:
        return 0.0

    return len(set(items)) / len(items)