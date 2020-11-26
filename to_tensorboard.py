import tensorflow as tf

g = tf.Graph()

with g.as_default() as g:
    tf.train.import_meta_graph('./.tmp/standalone.ckpt.meta')

    with tf.Session(graph=g) as sess:
        file_writer = tf.summary.FileWriter(logdir='checkpoint_log_dir/arcface50', graph=g)

