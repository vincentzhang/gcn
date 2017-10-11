import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import os
import sys
from random import shuffle
import pdb

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


class Shape(object):
    def __init__(self, num_img, dataset_str, perm=True):
        self._cur_train = 0
        self._cur_val = 0
        self._num = num_img
        #self._idx = list(range(self._num))
        self._dataset_str = dataset_str
        self._dir_path = os.path.dirname(os.path.realpath(__file__))

        self._train_idx = parse_index_file("{}/data/{}_train.idx".format(self._dir_path, dataset_str))
        self._val_idx = parse_index_file("{}/data/{}_val.idx".format(self._dir_path, dataset_str))
        pdb.set_trace()
        if perm:
            shuffle(self._train_idx)
            #shuffle(self._val_idx)
        print("dir_path: {}".format(self._dir_path))
        print("Using dataset {}".format(dataset_str))

    def load_train(self):
        #with open("{}/data/circle_1.pkl".format(dir_path, dataset_str), 'rb') as f:
        with open("{}/data/{}_{}.pkl".format(self._dir_path, self._dataset_str, self._train_idx[self._cur_train]), 'rb') as f:
            if sys.version_info > (3, 0):
                x, y, graph = pkl.load(f, encoding='latin1')
            else:
                x, y, graph = pkl.load(f)
        #import pdb;pdb.set_trace()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        self._cur_train += 1
        if self._cur_train >= len(self._train_idx):
            self._cur_train = 0
        # change labels into one-hot vector representation
        num_classes = 2
        y = self.one_hot(y, num_classes)
        #features, labels, adjacency_matrix
        return x,y,adj

    def one_hot(self, vec, num_classes):
        # only works when num_classes = 2
        one_hot_vec = np.zeros( (vec.shape[0], num_classes) )
        one_hot_vec[ np.arange(vec.shape[0]), vec ] = 1
        return one_hot_vec


    def load_val(self):
        """ Load all validation data """
        adjs = []
        xs = []
        ys = []
        for idx in self._val_idx:
            with open("{}/data/{}_{}.pkl".format(self._dir_path, self._dataset_str, idx), 'rb') as f:
                if sys.version_info > (3, 0):
                    x, y, graph = pkl.load(f, encoding='latin1')
                else:
                    x, y, graph = pkl.load(f)
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
            adjs.append(adj)
            feature = sp.csr_matrix(x)
            feature = preprocess_features(feature, False)
            xs.append(feature)
            ys.append(self.one_hot(y, 2))
        #features, labels, adjacency_matrix
        #return xs,ys,adjs
        return adjs, xs, ys

    def load_train_batch(self, batch_size):
        """ Load the next batch for the shape data """
        #names = ['x', 'y', 'graph']
        features = []
        labels = []
        adjs = []
        for i in range(batch_size):
            # feature: m x d, m: num of nodes, d: num of features
            # label: m x 1, (binary class)
            # adj: m x m
            feature, label, adj = self.load_train()
            # normalize the feature for each node by dividing the sum of all features
            feature = sp.csr_matrix(feature)
            feature = preprocess_features(feature, False)
            features.append(feature)
            labels.append(label)
            adjs.append(adj)

        #import pdb
        #pdb.set_trace()
        #test_idx_reorder = parse_index_file("{}/data/ind.{}.test.index".format(dir_path, dataset_str))
        #test_idx_range = np.sort(test_idx_reorder)

        #features = sp.vstack((allx, tx)).tolil()
        #features[test_idx_reorder, :] = features[test_idx_range, :]

        #labels = y
        #labels = np.vstack((ally, ty))
        #labels[test_idx_reorder, :] = labels[test_idx_range, :]

        #idx_test = test_idx_range.tolist()
        #idx_train = range(len(y))
        #idx_val = range(len(y), len(y)+500)

        #train_mask = sample_mask(idx_train, labels.shape[0])
        #val_mask = sample_mask(idx_val, labels.shape[0])
        #test_mask = sample_mask(idx_test, labels.shape[0])

        #y_train = np.zeros(labels.shape)
        #y_val = np.zeros(labels.shape)
        #y_test = np.zeros(labels.shape)
        #y_train[train_mask, :] = labels[train_mask, :]
        #y_val[val_mask, :] = labels[val_mask, :]
        #y_test[test_mask, :] = labels[test_mask, :]

        return adjs, features, labels
        #return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i in range(len(names)):
        with open("{}/data/ind.{}.{}".format(dir_path, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))


    print("For dataset {}".format(dataset_str))
    print([objects[i].shape for i in range(6)])

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    #import pdb;pdb.set_trace()
    test_idx_reorder = parse_index_file("{}/data/ind.{}.test.index".format(dir_path, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    #import pdb; pdb.set_trace()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation.

        returns:
            sparse matrix in COOrdinate format.
            sparse_mx: [(row,col), data, shape]
    """
    def to_tuple(mx):
        #import pdb;pdb.set_trace()
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features, normalize=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    if normalize:
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    else:
        features = features/300
    #import pdb;pdb.set_trace()
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    #import pdb;pdb.set_trace()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def construct_shape_feed_dict(features, support, labels, placeholders):
    """Construct feed dictionary.

        features: a list of features for a batch of images
        support:  a list of adjacency matrices for a batch of images
    """

    feed_dict = dict()
   # import pdb;pdb.set_trace()
    #feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels'][i]: labels[i] for i in range(len(labels))})
    feed_dict.update({placeholders['features'][i]: features[i] for i in range(len(features))})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    #feed_dict.update({placeholders['num_features_nonzero']: features[0][1].shape})
    feed_dict.update({placeholders['num_features_nonzero'][i]: features[i][1].shape for i in range(len(features))})
    #feed_dict.update({placeholders['lr']: })
    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
