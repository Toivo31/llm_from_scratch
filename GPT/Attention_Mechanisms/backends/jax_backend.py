import jax
from jax import numpy as jnp

class jax_backend:

    def __init__(self,vocal=True):
        if vocal == True:

            meth = [attr for attr in dir(self) if callable(getattr(self, attr)) and not attr.startswith("__")]
            print("Wrapped functions:",meth)

            pass

    def matmul(self,u,v):
        return jnp.matmul(u,v)
    
    def transpose(self,v):
        return jnp.transpose(v)
    
    def sqrt(self,x):
        return jnp.sqrt(x)
    
    def softmax(self,v):
        return jax.nn.softmax(v)
    
    def where(self,v):
        return jnp.where(v)
    
    def matrix_exp(self,v):
        return jnp.exp(v)
    
    def shape_last_dim(self,v): #might not work...
        return v.shape[-1]
    
 