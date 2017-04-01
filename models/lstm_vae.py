from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from synthetic_dataset import SyntheticDataset

z_dim = 5
h_dim = 5
embedding_size = 2
keep_prob = 0.9

batch_size = 1
max_length = 16

vocab_size = 16
start_of_sequence_id = 0
end_of_sequence_id = vocab_size - 1


embeddings = tf.get_variable(
    "embeddings", shape=[vocab_size, embedding_size], dtype=tf.float32)

# [B,L]
sentence = tf.placeholder(tf.int64, shape=[batch_size, None])
batch_sequence_lengths = tf.placeholder(tf.int32, shape=[batch_size])

# [B, L, D]
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

        # [B, Z]
        z_mu = tf.contrib.layers.fully_connected(
            inputs=last_layer_h,
            num_outputs=z_dim,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())

        # [B, Z]
        z_logvar = tf.contrib.layers.fully_connected(
            inputs=last_layer_h,
            num_outputs=z_dim,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())

        return z_mu, z_logvar


def sample_z(mu, log_var):
    """Sample a z-vector given parameters"""
    eps = tf.random_normal(shape=tf.shape(mu), stddev=1e-3)
    return mu + tf.exp(log_var / 2) * eps

# =============================== P(X|z) ======================================


with tf.variable_scope("generator"):
    decoder_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        h_dim, state_is_tuple=True)

    if (keep_prob < 1.0):
        decoder_lstm_cell = tf.contrib.rnn.DropoutWrapper(
            decoder_lstm_cell, output_keep_prob=keep_prob)

    W_softmax = tf.get_variable(
        name="softmax_weights",
        shape=[h_dim, vocab_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b_softmax = tf.get_variable(
        name="softmax_biases",
        shape=[vocab_size],
        initializer=tf.zeros_initializer())


def G_train(z):
    with tf.variable_scope("generator"):
        decoder_initial_h = tf.contrib.layers.fully_connected(
            inputs=z,
            num_outputs=h_dim,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())

        initial_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            tf.zeros(shape=[batch_size, h_dim]),
            decoder_initial_h)

        training_fn = tf.contrib.seq2seq.simple_decoder_fn_train(
            encoder_state=initial_state_tuple)

        outputs, final_state, final_context = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=decoder_lstm_cell,
            decoder_fn=training_fn,
            inputs=X,
            sequence_length=batch_sequence_lengths)

        # Calculate logits and return output
        reshaped_output = tf.reshape(outputs, shape=[-1, h_dim])
        logits = tf.matmul(reshaped_output, W_softmax) + b_softmax
        logits = tf.reshape(logits, shape=[batch_size, -1, vocab_size])

        return logits, final_state, final_context


def G_prediction(z):
    with tf.variable_scope("generator") as varscope:
        # Reuse trained weights
        varscope.reuse_variables()

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
            """Used to convert cell outputs to logits"""
            return tf.matmul(x, W_softmax) + b_softmax

        inference_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=initial_state_tuple,
            embeddings=embeddings,
            start_of_sequence_id=start_of_sequence_id,
            end_of_sequence_id=end_of_sequence_id,
            maximum_length=max_length,
            num_decoder_symbols=vocab_size,
            name="decoder_fn_inference")

        logits, final_state, final_context = tf.contrib.seq2seq.dynamic_rnn_decoder(
            cell=decoder_lstm_cell,
            decoder_fn=inference_fn,
            inputs=None,
            sequence_length=None)

        return logits, final_state, final_context

# =============================== TRAINING ====================================


z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
outputs, _, _ = G_train(z_sample)

# Reconstruction loss
loss_mask = tf.sequence_mask(
    batch_sequence_lengths, batch_size, dtype=tf.float32)

recon_loss = tf.contrib.seq2seq.sequence_loss(
    outputs, sentence, weights=loss_mask)

# KL-Divergence loss
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)

# Combined VAE loss
vae_loss = tf.reduce_mean(recon_loss + 0.05 * kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

# Summaries to track
tf.summary.scalar('recon_loss', tf.reduce_mean(recon_loss))
tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
tf.summary.scalar('vae_loss', vae_loss)

# =============================== INFERENCE ===================================

z = tf.placeholder(tf.float32, shape=[batch_size, z_dim])

# Samples from random z
X_samples, _, _ = G_prediction(z)

# ================================ RUNNER =====================================

merged = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter("logs/exp", sess.graph)

dataset = SyntheticDataset(
    num_emb=vocab_size - 1,
    seq_length=max_length - 2,  # Reduce size to account for start, end token
    start_token=start_of_sequence_id)

for it in range(10000):
    # Prepend the start-of-sequence token before each sentence.
    X_mb = [[start_of_sequence_id] + dataset.get_random_sequence() + [end_of_sequence_id]
            for size in range(batch_size)]
    sequence_length = [len(seq) for seq in X_mb]
    batch_max = max(sequence_length)

    # Padding to make sure all inputs have the same length.
    X_mb = [list(seq) + ([0] * (batch_max - len(seq))) for seq in X_mb]

    test_output, test_z, test_z_mu, test_z_logvar, _, loss, summary = sess.run(
        [outputs, z_sample, z_mu, z_logvar, solver, vae_loss, merged],
        feed_dict={sentence: X_mb, batch_sequence_lengths: sequence_length})

    summary_writer.add_summary(summary, it)

    if it % 100 == 0:
        test_output = np.argmax(test_output, axis=2)
        print('Epoch = {}'.format(it))
        print('Input = {}'.format(X_mb))
        print('Output = {}'.format(test_output))

        print('z_sample = {}'.format(test_z))
        print('z_mu = {}'.format(test_z_mu))
        print('z_logvar = {}'.format(test_z_logvar))

        # Sample a new z
        # test_z = np.random.randn(batch_size, z_dim)
        samples = sess.run(X_samples, feed_dict={z: test_z})
        targets = np.argmax(samples, axis=2)
        print('test_z = {}'.format(test_z))
        print('Sample = {}'.format(targets))

        print('Loss: {}'.format(loss))
        print()
