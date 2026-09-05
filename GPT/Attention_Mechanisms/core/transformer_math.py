
def dot_product_attention(q,k,v,backend, mask =None):
    
    d = backend.shape_last_dim(k)
    scores = backend.matmul(q,backend.transpose(k))/backend.sqrt(d)

    if mask is not None:
        scores = backend.where(mask,scores, -1e9)

    weights = backend.softmax(scores)
    attn_values = backend.matmul(weights,v)
    return attn_values


def GPT_multihead_attention(q,k,v,backend):
    d = backend.shape_last_dim(q)
    
    scores = backend.matmul(q,backend.transpose_batch(k))/backend.sqrt(d)
    weights = backend.softmax(scores)
    
    attn_values = backend.matmul(weights,v)
    return attn_values

        

    
        
    




### Orhogonal Self-Attentio (OSA) implemented naively from Zhang and Martens et al Feb 2026. 


def orthogonal_attention(q,k,v,backend,alpha):

    d = q.shape[-1]
    S = alpha / backend.sqrt(d) * (backend.matmul(q, backend.transpose(k)) - backend.matmul(k, backend.transpose(q)))
    Attention_X = backend.matrix_exp(S) 
    Attn_values = backend.matmul(Attention_X,v)

    return Attn_values #Note that OSA is non-causal, Like its noted in the paper above

