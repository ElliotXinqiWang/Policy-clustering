import jax
import jax.numpy as jnp
import numpy as np

def u(s,a):
    s=s.astype(jnp.int32).reshape(s.shape[0]*s.shape[1],-1)
    a=a.astype(jnp.int32).reshape(a.shape[0]*a.shape[1],-1)
    s = jnp.concatenate([s, a], axis=-1)

    def hash(x: jnp.ndarray, base: int = 131, mod: int = 2147483629):
        x = x.astype(jnp.int32)

        def scan_fn(carry, xi):
            carry = (carry * base + xi) % mod
            return carry, 0

        final_hash, _ = jax.lax.scan(scan_fn, init=jnp.int32(0), xs=x)
        return final_hash
    
    s = jax.vmap(hash)(s)
    return jnp.unique(s).shape[0]

def g(r,gamma):
    return jnp.mean(jnp.sum(axis=-1,a=r*(gamma**jnp.arange(r.shape[-1]))))