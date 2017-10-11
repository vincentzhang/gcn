from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

import sys
path = '.'
if path not in sys.path:
    sys.path.append(path)

from gcn.utils import *
from gcn.models import GCN, MLP

import pdb
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'circle', 'Dataset string.')  # 'syn', 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 500, 'Number of units in hidden layer 1.')
#flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
#flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 2000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
batch_size = 2
# init the shape class
shape = Shape(1000, FLAGS.dataset)
adjs, features, y_trains = shape.load_train_batch(batch_size)
pdb.set_trace()

# Some preprocessing
if FLAGS.model == 'gcn':
    support = []
    for adj in adjs:
        support.append(preprocess_adj(adj))
    num_supports = 1 # batch size in this case
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# for lr decay
global_step = tf.Variable(0, trainable=False)
boundaries = [1000, 1500]
values = [0.01, 0.02, 0.01]#0.005, 0.001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)# Later, whenever we perform an optimization step, we increment global_step.
increment_global_step_op = tf.assign_add(global_step, 1)


# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32, name="support") for _ in range(len(support))],
    'features': [tf.sparse_placeholder(tf.float32, shape=tf.constant(features[0][2], dtype=tf.int64), name="features") for _ in range(len(features))],
    'labels': [tf.placeholder(tf.float32, shape=(None,y_trains[0].shape[1]), name="labels") for _ in range(len(y_trains))],
    'dropout': tf.placeholder_with_default(0., shape=(), name="dropout"),
    'lr': learning_rate,
    'num_features_nonzero': [tf.placeholder(tf.int32, name="num_features_nonzero") for _ in range(len(features)) ] # helper variable for sparse dropout
}

adjs_val, features_val, y_vals = shape.load_val()
support_val = []
for adj in adjs_val:
    support_val.append(preprocess_adj(adj))
pdb.set_trace()
#placeholders_val = {
#    'support': [tf.sparse_placeholder(tf.float32, name="support_val") for _ in range(len(support_val))],
#    'features': [tf.sparse_placeholder(tf.float32, shape=tf.constant(features_val[0][2], dtype=tf.int64), name="features_val") for _ in range(len(features_val))],
#    'labels': [tf.placeholder(tf.float32, shape=(None, y_vals[0].shape[1]), name="labels") for _ in range(len(y_vals))],
#    'lr': learning_rate,
#    'dropout': tf.placeholder_with_default(0., shape=(), name="dropout_val"),
#    'num_features_nonzero': [tf.placeholder(tf.int32, name="num_features_nonzero") for _ in range(len(features_val)) ] # helper variable for sparse dropout
#}

# Create model
model = model_func(placeholders, input_dim=features[0][2][1], logging=True)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config) # allow the gpu memory to grow such that it doesn't take all the memory

# Define model evaluation function
#def evaluate(features, support, labels, mask, placeholders):
def evaluate(features, support, labels, placeholders):
    t_test = time.time()
    #pdb.set_trace()
    feed_dict_val = construct_shape_feed_dict(features, support, labels, placeholders)
    feed_dict_val.update({placeholders['dropout']: FLAGS.dropout})
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
#loss_val = []
label_trains = np.argmax(y_trains[0], 1)
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    #adjs, features, y_trains = shape.load_train_batch(batch_size)

    # Construct feed dictionary
    #feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    # features is a vector in this case
    #pdb.set_trace()
    feed_dict = construct_shape_feed_dict(features, support, y_trains, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    #feed_dict.update({placeholders['lr']: learning_rate})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, increment_global_step_op, model.prediction(), model.lr], feed_dict=feed_dict)
    #loss_val.append(outs[1])

    # Validation
    adjs_val, features_val, y_vals = shape.load_val()
    support_val = []
    for adj in adjs_val:
        support_val.append(preprocess_adj(adj))

    cost, acc, duration = evaluate(features_val, support_val, y_vals, placeholders)
    cost_val.append(cost)

    # Print results
    #print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1][0,0]))
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t), "lr={:.5f}".format(outs[-1]))
          #"val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t), "step={:d}".format(outs[-2]), "lr={:.5f}".format(outs[-1]))
    pdb.set_trace()

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
#test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
#print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

sess.close()
