import random
import math
import numpy as np
import time

import nearest_neighbors.octree as octree
import nearest_neighbors.kdtree as kdtree
from nearest_neighbors.result_set import KNNResultSet, RadiusNNResultSet


def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    iteration_num = 100

    print("octree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        db_np = (np.random.rand(db_size, dim) - 0.5) * 100

        begin_t = time.time()
        root = octree.octree_construction(db_np, leaf_size, min_extent)
        construction_time_sum += time.time() - begin_t

        query = np.random.rand(3)

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        octree.octree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search_fast(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
                                                                     knn_time_sum*1000/iteration_num,
                                                                     radius_time_sum*1000/iteration_num,
                                                                     brute_time_sum*1000/iteration_num))

    print("kdtree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        db_np = np.random.rand(db_size, dim)

        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t

        query = np.random.rand(3)

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))



if __name__ == '__main__':
    main()