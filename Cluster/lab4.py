import numpy as np
import math


def euclidean_distance(a, b):
    """Вычисляет евклидово расстояние между двумя точками"""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def kmeans_clustering(X, K=2, initial_centroids=None, verbose=True):
    n_samples = len(X)

    if initial_centroids is None:
        centroids = X[:K].copy()
    else:
        centroids = np.array(initial_centroids, dtype=float)

    iteration = 0

    if verbose:
        print(f"{'=' * 60}")
        print(f"Начальные центроиды:")
        for i, c in enumerate(centroids):
            print(f"  m_{i + 1}^{{(0)}} = [{c[0]:.1f}, {c[1]:.1f}]")
        print()

    while True:
        iteration += 1

        clusters = [[] for _ in range(K)]
        distances_log = []

        for i, x in enumerate(X):
            distances = [euclidean_distance(x, centroids[j]) for j in range(K)]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)
            distances_log.append((i + 1, x, distances, cluster_idx + 1))

        if verbose:
            print(f"--- Итерация {iteration} ---")
            print("Расстояния и назначения:")
            for idx, x, dists, assigned in distances_log:
                dist_str = ", ".join([f"d(X_{idx}, m_{j + 1}) = {d:.3f}" for j, d in enumerate(dists)])
                print(f"  X_{idx} = [{x[0]:.1f}, {x[1]:.1f}]: {dist_str} -> G{assigned}")
            print()

            print("Текущие кластеры:")
            for i, cluster in enumerate(clusters):
                points_str = ", ".join([f"X_{j + 1}" for j in cluster])
                print(f"  G{i + 1} = {{{points_str}}}")
            print()

        new_centroids = np.zeros_like(centroids)
        for i in range(K):
            if len(clusters[i]) > 0:
                cluster_points = X[clusters[i]]
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = centroids[i]

        if verbose:
            print("Новые центроиды:")
            for i, c in enumerate(new_centroids):
                if len(clusters[i]) > 0:
                    points = X[clusters[i]]
                    mean_str = " + ".join([f"[{p[0]:.1f}, {p[1]:.1f}]" for p in points])
                    print(f"  m_{i + 1}^{{({iteration})}} = [{c[0]:.3f}, {c[1]:.3f}] = mean({mean_str})")
                else:
                    print(f"  m_{i + 1}^{{({iteration})}} = [{c[0]:.3f}, {c[1]:.3f}] (без изменений)")
            print()

        if np.allclose(centroids, new_centroids):
            if verbose:
                print(f"Центроиды не изменились. Алгоритм сошелся за {iteration} итераций.")
                print(f"{'=' * 60}\n")
            break

        centroids = new_centroids

    final_clusters = [[] for _ in range(K)]
    for i, x in enumerate(X):
        distances = [euclidean_distance(x, centroids[j]) for j in range(K)]
        cluster_idx = np.argmin(distances)
        final_clusters[cluster_idx].append(i)

    return centroids, final_clusters


# ============================================================
# Все 13 наборов данных из задания
# ============================================================

datasets = [
    # 1
    np.array([[0, 3], [1, 1], [1, 0], [5, 5], [3, 4], [4, 3]]),
    # 2
    np.array([[0, 0], [8, 9], [2, 3], [6, 9], [1, 2], [8, 7]]),
    # 3
    np.array([[7, 5], [6, 4], [3, 4], [2, 1], [0, 7], [9, 9]]),
    # 4
    np.array([[4, 3], [2, 5], [1, 1], [2, 2], [0, 9], [8, 0]]),
    # 5
    np.array([[3, 3], [1, 4], [2, 1], [2, 7], [1, 9], [5, 8]]),
    # 6
    np.array([[7, 1], [0, 3], [2, 1], [8, 2], [1, 3], [0, 0]]),
    # 7
    np.array([[3, 8], [7, 1], [4, 2], [6, 0], [5, 2], [5, 0]]),
    # 8
    np.array([[4, 6], [4, 2], [8, 4], [4, 8], [3, 7], [7, 3]]),
    # 9
    np.array([[2, 1], [0, 5], [1, 5], [5, 0], [4, 6], [3, 2]]),
    # 10
    np.array([[4, 4], [3, 4], [9, 9], [8, 9], [9, 8], [7, 6]]),
    # 11
    np.array([[4, 4], [4, 4], [3, 7], [6, 6], [3, 1], [2, 3]]),
    # 12
    np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]),
    # 13
    np.array([[0, 1], [7, 0], [3, 7], [1, 9], [2, 8], [9, 2]]),
]

# Решение всех задач
for i, X in enumerate(datasets, 1):
    print(f"\n{'#' * 70}")
    print(f"ЗАДАЧА {i}")
    print(f"{'#' * 70}")
    print(f"Данные X = {X.tolist()}")
    print()

    centroids, clusters = kmeans_clustering(X, K=2)

    print("ИТОГОВЫЙ РЕЗУЛЬТАТ:")
    print(f"Центроиды: m1 = [{centroids[0][0]:.3f}, {centroids[0][1]:.3f}], "
          f"m2 = [{centroids[1][0]:.3f}, {centroids[1][1]:.3f}]")
    print("Кластеры:")
    for j, cluster in enumerate(clusters):
        points = [f"X_{k + 1}" for k in cluster]
        print(f"  G{j + 1} = {{{', '.join(points)}}}")