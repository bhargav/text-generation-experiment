"""
Variational Autoencoder for text generation using an LSTM as the encoder
and the decoder network
"""
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from readers.synthetic_dataset import SyntheticDataset


class VAE_LSTM_Model(object):
    """docstring for VAE_LSTM_Model"""

    def __init__(self, **kwargs):
        super(VAE_LSTM_Model, self).__init__()
        self.batch_size = batch_size = 10
        self.z_dim = 5
        self.vocab_size = 16
        self.embedding_size = 2
        self.start_of_sequence_id = 0
        self.end_of_sequence_id = 15
        self.max_length = 16

        # Encoder parameters
        self.enc_hidden_dims = 5
        self.enc_keep_prob = 0.9

        # Decoder parameters
        self.dec_hidden_dims = 10
        self.dec_keep_prob = 0.7

        # [batch_size, max_sentence_length]
        self.X = tf.placeholder(tf.int64, shape=[batch_size, None])
        self.batch_sequence_lengths = tf.placeholder(
            tf.int32, shape=[batch_size])

        self.build_graph()

    def build_graph(self):
        self.build_encoder_network()

        self.z = self.sample_z(self.z_mu, self.z_logvar)

        self.build_decoder_network()
        self.build_inference()
        self.setup_loss_and_train()

    def build_encoder_network(self):
        with tf.variable_scope("encoder"):
            lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(
                self.enc_hidden_dims, state_is_tuple=True)

            lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(
                self.enc_hidden_dims, state_is_tuple=True)

            if (self.enc_keep_prob < 1.0):
                lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell_fw, output_keep_prob=self.enc_keep_prob)

                lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell_bw, output_keep_prob=self.enc_keep_prob)

            # lstm_initial_state = lstm_cell_fw.zero_state(
            # self.batch_size, tf.float32)

            self._enc_embeddings = tf.get_variable(
                "embeddings",
                shape=[self.vocab_size, self.embedding_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer())
            encoder_input = tf.nn.embedding_lookup(
                self._enc_embeddings, ids=self.X)

            _, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell_fw,
                cell_bw=lstm_cell_bw,
                inputs=encoder_input,
                sequence_length=self.batch_sequence_lengths,
                dtype=tf.float32)

            self._enc_final_state = states

            # State returned is a tuple of (c, h) for the LSTM Cell
            last_layer_h = (states[0][1] + states[1][1]) / 2.

            # [B, Z]
            self.z_mu = tf.contrib.layers.fully_connected(
                inputs=last_layer_h,
                num_outputs=self.z_dim,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer())

            # [B, Z]
            self.z_logvar = tf.contrib.layers.fully_connected(
                inputs=last_layer_h,
                num_outputs=self.z_dim,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer())

    def build_decoder_network(self):
        with tf.variable_scope("decoder"):
            decoder_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.dec_hidden_dims, state_is_tuple=True)

            if (self.dec_keep_prob < 1.0):
                decoder_lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    decoder_lstm_cell, output_keep_prob=self.dec_keep_prob)

            self._dec_lstm_cell = decoder_lstm_cell

            decoder_initial_h = tf.contrib.layers.fully_connected(
                inputs=self.z,
                num_outputs=self.dec_hidden_dims,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer())

            initial_state_tuple = tf.contrib.rnn.LSTMStateTuple(
                tf.zeros(shape=[self.batch_size, self.dec_hidden_dims]),
                decoder_initial_h)
            self._dec_initial_state_tuple = initial_state_tuple

            self._dec_embeddings = tf.get_variable(
                "embeddings",
                shape=[self.vocab_size, self.embedding_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer())

            decoder_input = tf.nn.embedding_lookup(
                self._dec_embeddings, ids=self.X)

            outputs, final_state = tf.nn.dynamic_rnn(
                cell=decoder_lstm_cell,
                inputs=decoder_input,
                sequence_length=self.batch_sequence_lengths,
                initial_state=initial_state_tuple)

            self._dec_W_softmax = tf.get_variable(
                name="softmax_weights",
                shape=[self.dec_hidden_dims, self.vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())
            self._dec_b_softmax = tf.get_variable(
                name="softmax_biases",
                shape=[self.vocab_size],
                initializer=tf.zeros_initializer())

            # Calculate logits and return output
            reshaped_output = tf.reshape(
                outputs, shape=[-1, self.dec_hidden_dims])
            logits = tf.matmul(reshaped_output,
                               self._dec_W_softmax) + self._dec_b_softmax
            logits = tf.reshape(
                logits, shape=[self.batch_size, -1, self.vocab_size])

            self._dec_outputs = outputs
            self._dec_logits = logits
            self._dec_final_state = final_state

    def build_inference(self):
        with tf.variable_scope("decoder", reuse=True):

            def output_fn(x):
                """Used to convert cell outputs to logits"""
                return tf.matmul(x, self._dec_W_softmax) + self._dec_b_softmax

            inference_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=self._dec_initial_state_tuple,
                embeddings=self._dec_embeddings,
                start_of_sequence_id=self.start_of_sequence_id,
                end_of_sequence_id=self.end_of_sequence_id,
                maximum_length=self.max_length,
                num_decoder_symbols=self.vocab_size,
                name="decoder_fn_inference")

            logits, final_state, final_context = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=self._dec_lstm_cell,
                decoder_fn=inference_fn,
                inputs=None,
                sequence_length=None)

            self._inf_logits = logits
            self._inf_final_state = final_state

    def sample_z(self, mu, log_var):
        """Sample a z-vector given parameters"""
        eps = tf.random_normal(shape=tf.shape(mu), stddev=1e-3)
        # return mu + tf.exp(log_var / 2) * eps
        return mu + eps

    def setup_loss_and_train(self):
        """Returns the loss op"""
        with tf.variable_scope("loss"):
            # Reconstruction loss
            loss_mask = tf.sequence_mask(
                self.batch_sequence_lengths, self.max_length, dtype=tf.float32)

            recon_loss = tf.contrib.seq2seq.sequence_loss(
                self._dec_logits, self.X, weights=loss_mask)

            # KL-Divergence loss
            kl_loss = 0.5 * tf.reduce_sum(
                tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)

            # Combined VAE loss
            vae_loss = tf.reduce_mean(recon_loss) + 0.003 * tf.nn.l2_loss(
                self.z)

            # Summaries to track
            tf.summary.scalar('reconstruction_loss',
                              tf.reduce_mean(recon_loss))
            tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
            tf.summary.scalar('vae_loss', vae_loss)

            self._loss = vae_loss

        solver = tf.train.AdamOptimizer().minimize(self._loss)
        self._train_op = solver

    def train(self, session, input_batch):
        train_op = self.train_op()
        return session.run(train_op, feed_dict={self.X: input_batch})

    def predict(self, session, input_z):
        pass


if __name__ == "__main__":
    batch_size = 10
    start_of_sequence_id = 0
    end_of_sequence_id = 15
    vocab_size = 16
    max_length = 16

    dataset = SyntheticDataset(
        num_emb=15,
        seq_length=14,  # Reduce size to account for start, end token
        start_token=0)

    model = VAE_LSTM_Model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("logs/exp", sess.graph)

    for iteration in range(100000):
        X_batch = [[start_of_sequence_id] + dataset.get_random_sequence() +
                   [end_of_sequence_id] for size in range(batch_size)]
        sequence_length = [len(seq) for seq in X_batch]

        # Padding to make sure all inputs have the same length.
        X_mb = [list(seq) + ([0] * (max_length - len(seq))) for seq in X_batch]
        train_op = model._train_op

        dec_logits, loss, z_sample, z_mu, z_logvar, summary, _ = sess.run(
            [
                model._dec_logits, model._loss, model.z, model.z_mu,
                model.z_logvar, merged, train_op
            ],
            feed_dict={
                model.X: X_batch,
                model.batch_sequence_lengths: sequence_length
            })

        summary_writer.add_summary(summary, iteration)

        if iteration % 1000 == 0:
            dec_output = np.argmax(dec_logits, axis=2)
            print('Epoch = {0}, Loss = {1}'.format(iteration, loss))
            print('Input = {}'.format(X_batch))
            print('Output = {}'.format(dec_output))
            print('z_mean = {}'.format(z_mu))
            print('z_logvar = {}'.format(z_logvar))

            inf_logits = sess.run(
                model._inf_logits, feed_dict={model.z: z_sample})

            output_sample = np.argmax(inf_logits, axis=2)
            print('Output Sample = {}'.format(output_sample))
            print()
