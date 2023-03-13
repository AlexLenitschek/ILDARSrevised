def select_by_largest_cluster(clusters):
    assert len(clusters) > 0
    return [max(clusters, key=len)]
