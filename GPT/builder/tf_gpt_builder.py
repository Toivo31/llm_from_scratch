import tensorflow as tf

from  model_configs.config_reader import config

cfg=config("lobotomy_shakespeare.yaml")
model =cfg.model


from Attention_Mechanisms.core.transformer_math import dot_product_attention as attention
from Attention_Mechanisms.backends.tf_backend import tf_backend
from functools import partial

class n_Vaswani_Layer(tf.keras.layers.Layer):

    def __init__(self, *,num_heads,num_outputs, activity_regularizer = None, trainable = True, dtype = None, autocast = True, name = None, input_dim = None, input_shape = None):
        self.bknd = tf_backend(False)
        self.num_heads = num_heads
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, input_dim=input_dim, input_shape=input_shape)
        self.W_O = tf.keras.layers.Dense(num_outputs,dtype=tf.float32)
        self.head = partial(attention,backend=self.bknd)
        self.head_dim = num_outputs//num_heads
        self.W_q= []
        self.W_k= []
        self.W_v= []
        for __ in range(self.num_heads):
            self.W_q.append(tf.keras.layers.Dense(self.head_dim,dtype=tf.float32))
            self.W_k.append(tf.keras.layers.Dense(self.head_dim,dtype=tf.float32))
            self.W_v.append(tf.keras.layers.Dense(self.head_dim,dtype=tf.float32))



    
    def call(self,v):
        heads = []
        for index in range(self.num_heads):
            Q = self.W_q[index](v)
            K = self.W_k[index](v)
            V = self.W_v[index](v)

            heads.append(self.head(Q,K,V))


        attn_out = tf.concat(heads,axis=-1)
        output = self.W_O(attn_out)
        return output
 