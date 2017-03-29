import numpy as np
import tensorflow as tf
import random

z_dim = 16
h_dim = 16
keep_prob = 1.0
batch_size = 1
vocab_size = 32
start_of_sequence_id = 12
end_of_sequence_id = 29

max_length = 16
embedding_size = 128


embeddings = tf.get_variable(
    "embeddings", shape=[vocab_size, embedding_size], dtype=tf.float32)

sentence = tf.placeholder(tf.int32, shape=[batch_size, None])
batch_sequence_lengths = tf.placeholder(tf.int32, shape=[batch_size])
z = tf.placeholder(tf.float32, shape=[batch_size, z_dim])

X = tf.nn.embedding_lookup(embeddings, ids=sentence)

# =============================== Q(z|X) ======================================

with tf.variable_scope("encoder"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(h_dim, state_is_tuple=True)

    if (keep_prob < 1.0):
        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell, output_keep_prob=keep_prob)


def Q(X):
    with tf.variable_scope("encoder"):
        decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(
            encoder_state=lstm_cell.zero_state(batch_size, tf.float32))

        _, state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=lstm_cell,
            decoder_fn=decoder_fn,
            inputs=X,
            sequence_length=batch_sequence_lengths)

        # State returned is a tuple of (c, h) for the LSTM Cell
        last_layer_h = state[1]

        z_mu = tf.contrib.layers.fully_connected(
            inputs=last_layer_h,
            num_outputs=z_dim,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())

        z_logvar = tf.contrib.layers.fully_connected(
            inputs=last_layer_h,
            num_outputs=z_dim,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())

        return z_mu, z_logvar


def sample_z(mu, log_var):
    """Sample a z-vector given parameters"""
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

# =============================== P(X|z) ======================================


with tf.variable_scope("generator"):
    decoder_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        h_dim, state_is_tuple=True)

    if (keep_prob < 1.0):
        decoder_lstm_cell = tf.contrib.rnn.DropoutWrapper(
            decoder_lstm_cell, output_keep_prob=keep_prob)


def G(z, is_training):
    with tf.variable_scope("generator") as varscope:
        if not is_training:
            varscope.reuse_variables()

        # decoder_initial_c = tf.contrib.layers.fully_connected(
        #     inputs=z,
        #     num_outputs=h_dim,
        #     activation_fn=tf.nn.relu,
        #     weights_initializer=tf.contrib.layers.xavier_initializer(),
        #     biases_initializer=tf.zeros_initializer())

        decoder_initial_h = tf.contrib.layers.fully_connected(
            inputs=z,
            num_outputs=h_dim,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())

        initial_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            tf.zeros(shape=[batch_size, h_dim]),
            decoder_initial_h)

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

        if is_training:
            decoder_fn = training_fn
            inputs = X
            sequence_length = batch_sequence_lengths
        else:
            decoder_fn = inference_fn
            inputs = None
            sequence_length = None

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

# Reconstruction loss
batch_max_length = tf.shape(outputs)[1]
recon_loss = tf.contrib.seq2seq.sequence_loss(
    outputs, sentence, weights=tf.ones(shape=[batch_size, batch_max_length]))

# KL-Divergence loss
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar, 1)

# Combined VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

# Summaries to track
tf.summary.scalar('recon_loss', tf.reduce_mean(recon_loss))
tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
tf.summary.scalar('vae_loss', vae_loss)

merged = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter("logs/exp", sess.graph)

for it in range(10000):
    X_mb = [range(0, max_length) for _ in range(batch_size)]
    sequence_length = [len(seq) for seq in X_mb]
    batch_max = max(sequence_length)

    X_mb = [list(seq) + ([0] * (batch_max - len(seq))) for seq in X_mb]

    test_output, test_z, _, loss, summary = sess.run(
        [outputs, z_sample, solver, vae_loss, merged],
        feed_dict={sentence: X_mb, batch_sequence_lengths: sequence_length})

    if it % 1000 == 0:
        test_output = np.argmax(test_output, axis=2)
        print('Output = {}'.format(test_output))

        print('Loss: {}'.format(loss))
        summary_writer.add_summary(summary, it)

        # Sample a new z
        # test_z = np.random.randn(batch_size, z_dim)
        samples = sess.run(X_samples, feed_dict={z: test_z})
        targets = np.argmax(samples, axis=2)
        print('Sample = {}'.format(targets))
