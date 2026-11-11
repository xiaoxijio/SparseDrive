"""Compress all 900 anchor centers to 450, rather than truncating the file."""
import argparse
import os
import numpy as np
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/kmeans/kmeans_det_900.npy')
    parser.add_argument('--output', default='data/kmeans/kmeans_det_450.npy')
    parser.add_argument('--clusters', type=int, default=450)
    args = parser.parse_args()
    if os.path.exists(args.output):
        existing = np.load(args.output)
        if existing.shape != (args.clusters, 11):
            raise ValueError('Existing anchor file has unexpected shape')
        print('Keeping existing anchors:', args.output, existing.shape)
        return
    anchors = np.load(args.source)
    if anchors.shape[1] != 11 or len(anchors) < args.clusters:
        raise ValueError('Expected at least K detection anchors with 11 state values')
    centers = KMeans(n_clusters=args.clusters, random_state=0, n_init=10).fit(
        anchors[:, :3]).cluster_centers_
    # Same size/yaw/velocity initialization as the original kmeans_det.py.
    other = np.tile([1., 1., 1., 1., 0., 0., 0., 0.], (args.clusters, 1))
    result = np.concatenate([centers, other], axis=-1).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, result)
    print('Wrote', args.output, result.shape)


if __name__ == '__main__':
    main()
