from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.sparse import matmul


class SparseEmbedding(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SparseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=(self.output_dim, ),
                                    initializer='uniform',
                                    trainable=True)
        super(SparseEmbedding, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.bias_add(matmul(x, self.kernel), self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim
