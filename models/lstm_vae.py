import numpy as np
import tensorflow as tf
import random

z_dim = 100
h_dim = 128
keep_prob = 1.0
batch_size = 1
vocab_size = 1000
start_of_sequence_id = 12
end_of_sequence_id = 29

max_length = 16
embedding_size = 128


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


embeddings = tf.get_variable(
    "embeddings", shape=[vocab_size, embedding_size], dtype=tf.float32)

sent = tf.placeholder(tf.int32, shape=[None, max_length])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

X = tf.nn.embedding_lookup(embeddings, ids=sent)

# =============================== Q(z|X) ======================================

lstm_cell = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)

if (keep_prob < 1.0):
    lstm_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_cell, output_keep_prob=keep_prob)

Q_W_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(
        encoder_state=lstm_cell.zero_state(batch_size, tf.float32))

    sequence_lengths = [max_length] * batch_size

    # Get sequence lengths
    _, state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        cell=lstm_cell,
        decoder_fn=decoder_fn,
        inputs=X,
        sequence_length=sequence_lengths)

    last_layer_state = state[1]
    z_mu = tf.matmul(last_layer_state, Q_W_mu) + Q_b_mu
    z_logvar = tf.matmul(last_layer_state, Q_W_sigma) + Q_b_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    """Sample a z-vector given parameters"""
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

# =============================== P(X|z) ======================================


with tf.variable_scope("generator"):
    P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
    P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    decoder_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        h_dim, state_is_tuple=True)

    if (keep_prob < 1.0):
        decoder_lstm_cell = tf.contrib.rnn.DropoutWrapper(
            decoder_lstm_cell, output_keep_prob=keep_prob)


def G(z, is_training):
    with tf.variable_scope("generator") as varscope:
        if not is_training:
            varscope.reuse_variables()

        decoder_initial_state = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
        initial_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            tf.zeros(shape=[batch_size, h_dim], dtype=tf.float32),
            decoder_initial_state)

        def output_fn(x):
            return tf.contrib.layers.linear(x, vocab_size, scope=varscope)

        training_fn = tf.contrib.seq2seq.simple_decoder_fn_train(
            encoder_state=initial_state_tuple)

        # Should this be encoder's last state
        inference_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=initial_state_tuple,
            embeddings=embeddings,
            start_of_sequence_id=start_of_sequence_id,
            end_of_sequence_id=end_of_sequence_id,
            maximum_length=max_length,
            num_decoder_symbols=vocab_size,
            name="decoder_fn_inference")

        decoder_fn = training_fn if is_training else inference_fn
        inputs = X if is_training else None
        sequence_length = [max_length] * batch_size if is_training else None

        return tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=decoder_lstm_cell,
            decoder_fn=decoder_fn,
            inputs=inputs,
            sequence_length=sequence_length)

# =============================== TRAINING ====================================


z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
outputs, _, _ = G(z_sample, is_training=True)

# Samples from random z
X_samples, _, _ = G(z, is_training=False)

recon_loss = tf.contrib.seq2seq.sequence_loss(
    outputs, sent, weights=tf.ones(shape=[batch_size, max_length]))

kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar, 1)

vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('recon_loss', tf.reduce_mean(recon_loss))
tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
tf.summary.scalar('vae_loss', vae_loss)

summary_writer = tf.summary.FileWriter("logs/exp", sess.graph)
merged = tf.summary.merge_all()

for it in range(5000):
    # X_mb = [[random.randrange(vocab_size) for _ in range(max_length)]]
    X_mb = [range(max_length)]

    test_output, _, loss, summary = sess.run(
        [outputs, solver, vae_loss, merged], feed_dict={sent: X_mb})

    if it % 1000 == 0:
        test_output = np.argmax(test_output, axis=2)
        print('Output = {}'.format(test_output))

        print('Loss: {}'.format(loss))
        summary_writer.add_summary(summary, it)

        # Sample a new z
        samples = sess.run(X_samples, feed_dict={z: np.random.randn(1, z_dim)})
        targets = np.argmax(samples, axis=2)
        print('Output = {}'.format(targets))
