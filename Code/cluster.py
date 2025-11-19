from sklearn.cluster import DBSCAN
import numpy as np

def cluster_points(points, eps=10, min_samples=2):
    # Only use (x,y) for clustering
    p = np.array(points)
    xy = p[:, :2]

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
    labels = np.array(clustering.labels_)
    return labels


def extract_cluster_distances(points, labels):
    clusters = {}

    for label in set(labels):
        if label == -1:
            continue  # noise
        
        cluster_pts = points[labels == label]
        centroid = cluster_pts.mean(axis=0)    # (x_b, y_b, z_b)
        
        # Distance from robot (robot located at (0,0) in its own frame)
        distance = np.linalg.norm(centroid[:2])

        clusters[label] = {
            "centroid": centroid,
            "distance": distance
        }

    return clusters