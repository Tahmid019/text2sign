# cnn1dLstm.py
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense, Layer

# custom Attention layer
class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, time, features)
        e = tf.math.tanh(tf.tensordot(x, self.W, axes=[[2], [0]]))
        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(x * alpha, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# builder function for CNN+BiLSTM+Attention model

def build_cnn_lstm_attention(seq_len, feat_dim, num_classes):
    inputs = Input(shape=(seq_len, feat_dim))
    x = inputs

    # Conv block 1
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    # Conv block 2
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    # Bidirectional LSTM stack
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Attention aggregation
    x = Attention()(x)
    x = Dropout(0.3)(x)

    # Classification head
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model