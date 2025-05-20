from flax import linen as nn
import functools
from flax.linen.initializers import constant, orthogonal, normal, he_normal, he_uniform
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from typing import Sequence, Dict

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(carry.shape[0], carry.shape[1]),
            rnn_state,
        )
        # rnn_state = self.initialize_carry(carry.shape[0], carry.shape[1])
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size, seed=0):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(seed), (batch_size, hidden_size))
    
class NotScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(carry.shape[0], carry.shape[1]),
            rnn_state,
        )
        rnn_state = self.initialize_carry(carry.shape[0], carry.shape[1])
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size, seed=0):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(seed), (batch_size, hidden_size))
class DiscreteActorRNN(nn.Module):
    """
    Discrete actor network with RNN.
    Output a categorical distribution over actions.
    """
    action_dim: Sequence[int]
    config: Dict
    not_rnn: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        if len(x) == 3:
            obs, dones, avail_actions = x
        else:
            obs, dones = x
            avail_actions = jnp.ones((obs.shape[0], obs.shape[1], self.action_dim))
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        if self.not_rnn:
            hidden, embedding = NotScannedRNN()(hidden, rnn_in)
        else:
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi
    
    def _get_probability_of_traj(self, hidden, actions, obs, dones):
        raise NotImplementedError("This function is not implemented for the discrete actor.")
class ContinuousActorRNN(nn.Module):
    """
    Continuous actor network with RNN.
    Output a Gaussian distribution over actions.
    """
    action_dim: int
    hidden_dim: int
    config: Dict
    not_rnn: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        if len(x) == 3:
            obs, dones, avail_actions = x
        else:
            obs, dones = x
            avail_actions = jnp.ones((obs.shape[0], obs.shape[1], self.action_dim))
            
        embedding = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        # print("self.not_rnn", self.not_rnn)
        if self.not_rnn:
            hidden, embedding = NotScannedRNN()(hidden, rnn_in)
        else:
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_std = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        actor_std = jax.nn.softplus(actor_std) + 1e-5
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.MultivariateNormalDiag(loc=action_logits, scale_diag=actor_std)

        return hidden, pi
class ActorCriticRNN(nn.Module):
    action_dim: int
    config: Dict 
    
    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        
        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    
class ContinuousActorCriticRNN(nn.Module):
    action_dim: int
    config: Dict 
    
    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.relu(nn.Dense(
            128#, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs))
        embedding = nn.relu(nn.Dense(
            128#, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs))
        
        rnn_in = (embedding, dones)
        hidden, rnn_out = ScannedRNN()(hidden, rnn_in)
        
        embedding = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            rnn_out
        )
        embedding = nn.relu(embedding)
        actor_mean = nn.Dense(
            self.action_dim#, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        actor_std = nn.Dense(
            self.action_dim#, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        actor_std = jax.nn.softplus(actor_std) + 1e-5
        
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=actor_std)
        
        
        critic = nn.relu(nn.Dense(128#, kernel_init=orthogonal(2), bias_init=constant(0.0)
                                  )(rnn_out))
        critic = nn.relu(nn.Dense(128#, kernel_init=orthogonal(2), bias_init=constant(0.0)
                                  )(rnn_out))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
class Encoder(nn.Module):
    latent_dim: int  # Latent space dimension
    hidden_dim: int  # Hidden state dimension

    @nn.compact
    def __call__(self, x, act):
        obs, dones = x  # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size), act: (seq_len, batch_size) or (seq_len, batch_size, act_dim)
        act=act.reshape(act.shape[0],act.shape[1],-1)

        # Embedding layer
        embedding_obs = obs
        embedding_obs = nn.relu(nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding_obs))
        # embedding_obs = nn.relu(nn.Dense(128)(embedding_obs))
        
        embedding_act = act
        embedding_act = nn.relu(nn.Dense(32)(embedding_act))
        embedding_act = nn.relu(nn.Dense(128)(embedding_act))

        embedding = jnp.concatenate([embedding_obs, embedding_act], axis=-1)
        # embedding = embedding_obs
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        # RNN hidden state initialization
        batch_size = embedding.shape[1]
        hidden = ScannedRNN.initialize_carry(batch_size, self.hidden_dim)

        # RNN processing
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Index first 'done'
        first_done = jnp.argmax(dones, axis=0)
        first_done = jnp.where(jnp.any(dones, axis=0), first_done, obs.shape[0])

        # Select embeddings based on first_done
        batch_indices = jnp.arange(batch_size)
        needed_embedding = embedding[first_done-1, batch_indices]
        # needed_embedding = jnp.concatenate([needed_embedding,embedding[first_done//2, batch_indices],embedding[first_done*3//4, batch_indices]], axis=-1)
        # needed_embedding = nn.relu(nn.Dense(128)(needed_embedding))

        # Compute latent space parameters
        mu = nn.Dense(self.latent_dim)(needed_embedding)
        log_var = nn.Dense(self.latent_dim)(needed_embedding)
        return mu, log_var # (batch_size, latent_dim)
class Encoder_rnn_attention(nn.Module):
    latent_dim: int  # Hidden state dimension
    hidden_dim: int  # Hidden state dimension
    heads: int
    qk_dim: int = 1 # Latent space dimension
    # Q: (heads, qk_dim)
    # K: (seq_len, batch_size, heads, qk_dim)
    # V: (seq_len, batch_size, heads, latent_dim)


    def setup(self):
        self.Q = self.param('Q', nn.initializers.uniform(1), (self.heads, self.qk_dim))

    @nn.compact
    def __call__(self, x, act):
        obs, done = x  # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size), act: (seq_len, batch_size) or (seq_len, batch_size, act_dim)
        # print("act shape", act.shape)
        act=act.reshape(act.shape[0],act.shape[1],-1)
        done_mask = jnp.cumprod(1 - done.astype(jnp.int32), axis=0)
        seq_len = obs.shape[0]
        batch_size = obs.shape[1]

        # Embedding layer
        embedding_obs = obs
        embedding_obs = nn.relu(nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding_obs))
        # embedding_obs = nn.relu(nn.Dense(128)(embedding_obs))
        
        embedding_act = act
        embedding_act = nn.relu(nn.Dense(32)(embedding_act))
        embedding_act = nn.relu(nn.Dense(128)(embedding_act))

        embedding = jnp.concatenate([embedding_obs, embedding_act], axis=-1)
        # embedding = embedding_obs
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)


        hidden = ScannedRNN.initialize_carry(batch_size, self.hidden_dim)
        rnn_in = (embedding, done)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        embedding = nn.relu(embedding)

        K = nn.Dense(self.heads*self.qk_dim)(embedding).reshape(seq_len, batch_size, self.heads, self.qk_dim)
        K = nn.relu(K)
        V = nn.Dense(self.heads*self.latent_dim)(embedding).reshape(seq_len, batch_size, self.heads, self.latent_dim)

        K = K.transpose((1, 2, 0, 3))  # (batch_size, heads, seq_len, qk_dim)
        V = V.transpose((1, 2, 0, 3))  # (batch_size, heads, seq_len, latent_dim)
        A = (self.Q[None,:,None,:] * K).sum(axis=-1) / jnp.sqrt(self.qk_dim) # (batch_size, heads, seq_len)
        # jax.debug.print("A {}, d {}", A.shape, done_mask.shape)
        A = A * done_mask.transpose((1,0))[:, None, :]
        A = nn.softmax(A)
        result = (A[:,:,:,None] * V).sum(axis=2)  # (batch_size, heads, latent_dim)

        result = result.reshape(batch_size, -1)

        mu = nn.Dense(self.latent_dim)(result)
        log_var = nn.Dense(self.latent_dim)(result)
        return mu, log_var
class Encoder_self_attention(nn.Module):
    latent_dim: int  # Latent space dimension
    hidden_dim: int  # Hidden state dimension
    heads: int
    layers: int

    def setup(self):
        self.Q = self.param('Q', nn.initializers.uniform(1), (self.heads, self.hidden_dim))

    def self_attention(self, x, mask):
        # x: (seq_len, batch_size, latent_dim)
        # mask: (seq_len, batch_size )
        Q = nn.relu(nn.Dense(self.heads*self.hidden_dim)(x).reshape(x.shape[0], x.shape[1], self.heads, self.hidden_dim))
        K = nn.relu(nn.Dense(self.heads*self.hidden_dim)(x).reshape(x.shape[0], x.shape[1], self.heads, self.hidden_dim))
        V = nn.relu(nn.Dense(self.heads*self.latent_dim)(x).reshape(x.shape[0], x.shape[1], self.heads, self.latent_dim))
        Q = Q.transpose((1, 2, 0, 3))  # (batch_size, heads, seq_len, hidden_dim)
        K = K.transpose((1, 2, 0, 3))  # (batch_size, heads, seq_len, hidden_dim)
        V = V.transpose((1, 2, 0, 3))  # (batch_size, heads, seq_len, latent_dim)
        A = Q@K.transpose((0,1,3,2))/jnp.sqrt(self.hidden_dim) # (batch_size, heads, seq_len, seq_len)
        A = A * mask.transpose((1,0))[:, None, :, None]
        A = nn.softmax(A, axis=-1) # (batch_size, heads, seq_len, latent_dim)
        result = (A@V).transpose((2,0,1,3)).reshape(x.shape[0], x.shape[1], -1) # (seq_len, batch_size, heads*latent_dim)
        return nn.relu(nn.Dense(self.latent_dim)(result))

    @nn.compact
    def __call__(self, x, act):
        obs, done = x  # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size), act: (seq_len, batch_size) or (seq_len, batch_size, act_dim)
        act=act.reshape(act.shape[0],act.shape[1],-1)
        done_mask = jnp.cumprod(1 - done.astype(jnp.int32), axis=0)
        seq_len = obs.shape[0]
        batch_size = obs.shape[1]

        # Embedding layer
        embedding_obs = obs
        embedding_obs = nn.relu(nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding_obs))
        # embedding_obs = nn.relu(nn.Dense(128)(embedding_obs))
        embedding_act = act
        embedding_act = nn.relu(nn.Dense(32)(embedding_act))
        embedding_act = nn.relu(nn.Dense(128)(embedding_act))
        embedding = jnp.concatenate([embedding_obs, embedding_act], axis=-1)
        # embedding = embedding_obs
        embedding = nn.relu(nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding))

        attention = self.self_attention(embedding, done_mask)
        embedding = jax.lax.cond(self.layers>0, lambda _: attention, lambda _: embedding, None)
        attention = self.self_attention(embedding, done_mask)
        embedding = jax.lax.cond(self.layers>1, lambda _: attention, lambda _: embedding, None)
        attention = self.self_attention(embedding, done_mask)
        embedding = jax.lax.cond(self.layers>2, lambda _: attention, lambda _: embedding, None)

        # Q: (heads, hidden_dim)
        # K: (seq_len, batch_size, heads, hidden_dim)
        # V: (seq_len, batch_size, heads, latent_dim)
        K = nn.Dense(self.heads*self.hidden_dim)(embedding).reshape(seq_len, batch_size, self.heads, self.hidden_dim)
        K = nn.relu(K)
        V = nn.Dense(self.heads*self.latent_dim)(embedding).reshape(seq_len, batch_size, self.heads, self.latent_dim)

        K = K.transpose((1, 2, 0, 3))  # (batch_size, heads, seq_len, hidden_dim)
        V = V.transpose((1, 2, 0, 3))  # (batch_size, heads, seq_len, latent_dim)
        print("shape", K.shape, self.Q.shape)
        A = (self.Q[None,:,None,:] * K).sum(axis=-1) / jnp.sqrt(self.hidden_dim) # (batch_size, heads, seq_len)
        # jax.debug.print("A {}, d {}", A.shape, done_mask.shape)
        A = A * done_mask.transpose((1,0))[:, None, :]
        A = nn.softmax(A)
        result = (A[:,:,:,None] * V).sum(axis=2)  # (batch_size, heads, latent_dim)

        result = result.reshape(batch_size, -1)

        mu = nn.Dense(self.latent_dim)(result)
        log_var = nn.Dense(self.latent_dim)(result)
        return mu, log_var
class Decoder(nn.Module):
    action_dim: int  # action space dimension

    @nn.compact
    def __call__(self, z, x):
        obs, dones = x # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size)
        z = jnp.expand_dims(z, axis=0)  # z: (1, batch_size, latent_dim)
        z = jnp.broadcast_to(z, (obs.shape[0], *z.shape[1:]))  # z: (seq_len, batch_size, latent_dim)

        
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        embedding = jnp.concatenate([embedding, z], axis=-1)
        embedding = nn.sigmoid(nn.Dense(32)(embedding))
        embedding = nn.LayerNorm()(embedding)
        embedding = nn.sigmoid(nn.Dense(32)(embedding))
        # embedding = nn.LayerNorm()(embedding)
        
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        
        pi = distrax.Categorical(logits=actor_logits)
        return pi
class ContinuousDecoder(nn.Module):
    action_dim: int  # action space dimension

    @nn.compact
    def __call__(self, z, x):
        obs, dones = x # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size)
        z = jnp.expand_dims(z, axis=0)  # z: (1, batch_size, latent_dim)
        z = jnp.broadcast_to(z, (obs.shape[0], *z.shape[1:]))  # z: (seq_len, batch_size, latent_dim)

        
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        embedding = jnp.concatenate([embedding, z], axis=-1)
        actor_std = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        actor_std = jax.nn.softplus(actor_std) + 1e-5
        
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=actor_std)
        return pi
class VAE(nn.Module):
    latent_dim: int
    Encoder_hidden_dim: int
    action_dim: int
    discrete_action: bool
    
    attention: bool = False
    encoder_heads: int = 4

    def setup(self):
        if self.attention:
            self.encoder = Encoder_rnn_attention(self.latent_dim, self.Encoder_hidden_dim, self.encoder_heads)
        else:
            self.encoder = Encoder(self.latent_dim, self.Encoder_hidden_dim)
        self.decoder = Decoder(self.action_dim) if self.discrete_action else ContinuousDecoder(self.action_dim)

    def reparameterize(self, mu, log_var, rng):
        """Reparameterization trick."""
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, mu.shape)
        return mu + eps * std

    def __call__(self, x, act, rng):
        mu, log_var = self.encoder(x, act)  # Encode
        z = self.reparameterize(mu, log_var, rng)  # Reparameterization
        pi = self.decoder(z, x)  # Decode
        return pi, mu, log_var
class VQVAE(nn.Module):
    latent_dim: int
    Encoder_hidden_dim: int
    action_dim: int
    discrete_policy: bool
    attention: bool
    k: int
    alpha: float = 1
    beta: float = 0.25
    encoder_heads: int = 4
    pre_process: str = 'rnn'
    pre_process_layers: int = 1
    qk_dim: int = 1

    def setup(self):
        if self.attention:
            if self.pre_process == 'none':
                self.pre_process = "self_attention"
                self.pre_process_layers = 0
            if self.pre_process == 'rnn':
                self.encoder = Encoder_rnn_attention(self.latent_dim, self.Encoder_hidden_dim, self.encoder_heads, qk_dim=self.qk_dim)
            elif self.pre_process == 'self_attention':
                self.encoder = Encoder_self_attention(self.latent_dim, self.Encoder_hidden_dim, self.encoder_heads, self.pre_process_layers)
            else:
                raise ValueError(f"pre_process {self.pre_process} not supported")
        else:
            self.encoder = Encoder(self.latent_dim, self.Encoder_hidden_dim)
        if self.discrete_policy:
            self.decoder = Decoder(self.action_dim)
        else:
            self.decoder = ContinuousDecoder(self.action_dim)
        self.codebook = self.param('codebook', nn.initializers.normal(1), (self.k, self.latent_dim))

    def reparameterize(self, z):
        euc_dis = jnp.sum((z[:,None,:] - self.codebook[None,:,:])**2, axis=-1)
        z_q = jnp.argmin(euc_dis, axis=-1)
        z_q = self.codebook[z_q]
        loss = jnp.mean(jax.lax.stop_gradient(z_q) - z)**2 * self.alpha + \
               jnp.mean(z - jax.lax.stop_gradient(z_q))**2 * self.beta * self.alpha
        return z + jax.lax.stop_gradient(z_q - z), loss

    def __call__(self, x, act):
        mu, log_var = self.encoder(x, act)  # Encode
        z, loss = self.reparameterize(mu)  # Reparameterization
        print(mu.shape,z.shape)
        pi = self.decoder(z, x)  # Decode
        return pi, z, loss
class CAAE(nn.Module):
    latent_dim: int
    Encoder_hidden_dim: int
    action_dim: int
    discrete_policy: bool
    k: int
    alpha: float
    beta: float

    attention: bool
    encoder_heads: int = 4
    pre_process: str = 'rnn'
    pre_process_layers: int = 1
    gamma: float = 0

    use_sigma: bool = False

    method: str = 'max'

    def setup(self):
        if self.attention:
            if self.pre_process == 'none':
                self.pre_process = "self_attention"
                self.pre_process_layers = 0
            if self.pre_process == 'rnn':
                self.encoder = Encoder_rnn_attention(self.latent_dim, self.Encoder_hidden_dim, self.encoder_heads)
            elif self.pre_process == 'self_attention':
                self.encoder = Encoder_self_attention(self.latent_dim, self.Encoder_hidden_dim, self.encoder_heads, self.pre_process_layers)
            else:
                raise ValueError(f"pre_process {self.pre_process} not supported")
        else:
            self.encoder = Encoder(self.latent_dim, self.Encoder_hidden_dim)
        if self.discrete_policy:
            self.decoder = Decoder(self.action_dim)
        else:
            self.decoder = ContinuousDecoder(self.action_dim)

        # self.mu is the codebook
        self.mu = self.param('mu', nn.initializers.normal(1), (self.k, self.latent_dim))
        if self.use_sigma:
            self.sigma = self.param('sigma', lambda r: jnp.array(1.))
        else:
            self.sigma = jnp.array(1.)
        if self.method == 'max':
            self.method_id=0
        elif self.method == 'sum':
            self.method_id=1
        else:
            raise ValueError(f"method {self.method} not supported")
        # if self.use_sigma:
        #     def init_param(key,shape):
        #         k, d, _ = shape
        #         return jnp.tile(jnp.eye(d)[None,:,:],(k,1,1))#+jax.random.normal(key,(k,d,d))*0.001
        #     self.sigma = self.param('sigma', init_param, (self.k, self.latent_dim,self.latent_dim))
        # else:
        #     self.sigma = jnp.tile(jnp.eye(self.latent_dim)[None,:,:],(self.k,1,1))
        
    def log_pdf(self, x, mu): # return (batch_size, k)
        x=x[:,None,:]-mu[None,:,:]
        return -0.5*jnp.sum(x**2,axis=-1)*jnp.abs(self.sigma) - 0.5*jnp.log(jnp.abs(self.sigma)+1e-5)
        x=x[:,:,:,None]# (batch_size, k, latent_dim, 1)
        sigma=1/2*(self.sigma+self.sigma.transpose((0,2,1))) # make sure sigma is symmetric
        return -0.5*(x.transpose((0,1,3,2))@sigma@x).reshape(x.shape[0],x.shape[1])\
               -0.5*jnp.log(jnp.linalg.det(sigma)+1e-5)
    
    def sum(self, x):
        x=nn.relu(x)
        return jax.lax.cond(self.method_id==0,lambda _: jnp.max(x,axis=0),lambda _: (x*nn.softmax(x,axis=0)).sum(axis=0),None)

    def reparameterize(self, z):
        # pdf = self.log_pdf(z,self.mu)
        # loss = -jnp.max(pdf, axis=0).mean()*self.loss_weight
        loss = -self.sum(self.log_pdf(z,jax.lax.stop_gradient(self.mu))).mean() / (1+self.beta)
        loss+= -self.sum(self.log_pdf(jax.lax.stop_gradient(z),self.mu)).mean() * self.beta / (1+self.beta)

        kl = jnp.sum((self.mu[:,None,:] - self.mu[None,:,:])**2, axis=-1)
        return z, loss * self.alpha - jnp.minimum(kl,jnp.ones_like(kl)).mean() * self.gamma

    def __call__(self, x, act):
        mu, log_var = self.encoder(x, act)  # Encode
        z, loss = self.reparameterize(mu)  # Reparameterization
        # print(mu.shape,z.shape)
        pi = self.decoder(z, x)  # Decode
        return pi, z, loss
    
class CAAE_few_sample(nn.Module):
    latent_dim: int
    Encoder_hidden_dim: int
    action_dim: int
    discrete_policy: bool
    k: int
    alpha: float
    # beta: float # beta is set to 1

    attention: bool
    encoder_heads: int = 4
    pre_process: str = 'rnn'
    pre_process_layers: int = 1

    use_sigma: bool = False

    def setup(self):
        if self.attention:
            if self.pre_process == 'none':
                self.pre_process = "self_attention"
                self.pre_process_layers = 0
            if self.pre_process == 'rnn':
                self.encoder = Encoder_rnn_attention(self.latent_dim, self.Encoder_hidden_dim, self.encoder_heads)
            elif self.pre_process == 'self_attention':
                self.encoder = Encoder_self_attention(self.latent_dim, self.Encoder_hidden_dim, self.encoder_heads, self.pre_process_layers)
            else:
                raise ValueError(f"pre_process {self.pre_process} not supported")
        else:
            self.encoder = Encoder(self.latent_dim, self.Encoder_hidden_dim)
        if self.discrete_policy:
            self.decoder = Decoder(self.action_dim)
        else:
            self.decoder = ContinuousDecoder(self.action_dim)

        # self.mu is the codebook
        self.mu = self.param('mu', nn.initializers.normal(1), (self.k, self.latent_dim))
        if self.use_sigma:
            self.sigma = self.param('sigma', lambda r: jnp.array(1.))
        else:
            self.sigma = jnp.array(1.)
        # if self.use_sigma:
        #     def init_param(key,shape):
        #         k, d, _ = shape
        #         return jnp.tile(jnp.eye(d)[None,:,:],(k,1,1))#+jax.random.normal(key,(k,d,d))*0.001
        #     self.sigma = self.param('sigma', init_param, (self.k, self.latent_dim,self.latent_dim))
        # else:
        #     self.sigma = jnp.tile(jnp.eye(self.latent_dim)[None,:,:],(self.k,1,1))
        self.ln=nn.LayerNorm()
        
    def log_pdf(self, x, mu): # return (batch_size, k)
        x=x[:,None,:]-mu[None,:,:]
        return -0.5*jnp.sum(x**2,axis=-1)*jnp.abs(self.sigma) - 0.5*jnp.log(jnp.abs(self.sigma)+1e-5)
        x=x[:,:,:,None]# (batch_size, k, latent_dim, 1)
        sigma=1/2*(self.sigma+self.sigma.transpose((0,2,1))) # make sure sigma is symmetric
        return -0.5*(x.transpose((0,1,3,2))@sigma@x).reshape(x.shape[0],x.shape[1])\
               -0.5*jnp.log(jnp.linalg.det(sigma)+1e-5)
    
    def calc(self, z):
        mat=self.log_pdf(z,self.mu)
        sloss=self.log_pdf(self.mu,self.mu)
        sloss=jnp.maximum(sloss, -jnp.ones_like(sloss)).mean()
        return z, mat, sloss

    def __call__(self, x, act):
        mu, log_var = self.encoder(x, act)  # Encode
        mu=self.ln(mu)
        z, mat, sloss = self.calc(mu)  # Reparameterization
        # print(mu.shape,z.shape)
        pi = self.decoder(z, x)  # Decode
        # jax.debug.print("sloss {}", sloss)
        return pi, z, mat, sloss
class VQVAE_gumble_softmax(nn.Module):
    latent_dim: int
    Encoder_hidden_dim: int
    action_dim: int
    discrete_policy: bool
    k: int
    alpha: float = 1
    beta: float = 0.25

    def setup(self):
        self.encoder = Encoder(self.latent_dim, self.Encoder_hidden_dim)
        if self.discrete_policy:
            self.decoder = Decoder(self.action_dim)
        else:
            self.decoder = ContinuousDecoder(self.action_dim)
        self.codebook = self.param('codebook', nn.initializers.normal(1), (self.k, self.latent_dim))

    def temp(self,it):
        return 0.1+0.9*(0.95**it)

    def reparameterize(self, z, rng, it):
        euc_dis = -jnp.sum((z[:,None,:] - self.codebook[None,:,:])**2, axis=-1)
        # gumbel_noise = jax.random.gumbel(rng, shape=euc_dis.shape)
        # euc_dis = (euc_dis + gumbel_noise) / self.temp(it)
        euc_dis = euc_dis / self.temp(it)
        # z_e = jax.nn.softmax(euc_dis)
        z_e = jax.random.categorical(rng, euc_dis)
        z_q = jnp.argmax(z_e, axis=-1)
        z_q = self.codebook[z_q]
        loss = jnp.mean(jax.lax.stop_gradient(z_q) - z)**2 * self.alpha + \
               jnp.mean(z - jax.lax.stop_gradient(z_q))**2 * self.beta
        return z + jax.lax.stop_gradient(z_q - z), z_e, loss

    def __call__(self, x, act, rng, it):
        mu, log_var = self.encoder(x, act)  # Encode
        zq, ze, loss = self.reparameterize(mu, rng, it)  # Reparameterization
        print(mu.shape,zq.shape,ze.shape)
        pi = self.decoder(zq, x)  # Decode
        return pi, zq, ze, loss
class EncoderWrapper(nn.Module):
    latent_dim: int
    hidden_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim, self.hidden_dim)

    def __call__(self, x):
        return self.encoder(x)
class ClusteringLayer(nn.Module):
    n_clusters: int
    latent_dim: int

    def setup(self):
        self.centers = self.param('centers', jax.nn.initializers.normal(), (self.n_clusters, self.latent_dim))

    """
    Input: z  (batch_size, latent_dim)
    Output: q (batch_size, n_clusters)
    """
    def __call__(self, z):
        # 计算 q_ij
        q = 1.0 / (1.0 + jnp.sum((z[:, None, :] - self.centers[None, :, :]) ** 2, axis=2))
        q = q / jnp.sum(q, axis=1, keepdims=True)
        return q
class DEC(nn.Module):
    latent_dim: int
    n_clusters: int
    action_dim: int
    discrete_action: bool
    Encoder_hidden_dim: int = 32
    
    attention: bool = False
    encoder_heads: int = 4
    qk_dim: int = 1

    def setup(self):
        if self.attention:
            self.encoder = Encoder_rnn_attention(self.latent_dim, self.Encoder_hidden_dim, self.encoder_heads, qk_dim=self.qk_dim)
        else:
            self.encoder = Encoder(self.latent_dim, self.Encoder_hidden_dim)
        self.cluster_layer = ClusteringLayer(self.n_clusters, self.latent_dim)
        self.decoder = Decoder(self.action_dim) if self.discrete_action else ContinuousDecoder(self.action_dim)

    def encode(self, x):
        return self.encoder.encoder(x)[0]

    def __call__(self, x, act, rng):
        z = self.encoder(x, act)[0]
        q = self.cluster_layer(z)
        x_hat = self.decoder(z, x)
        return x_hat, q, z

class LoRA_mod(nn.Module):
    # d: int # input dim
    r: int # hidden dim
    k: int # output dim
    m: int # number of policies

    @nn.compact
    def __call__(self, x, id): # x: (seq_len, batch_size, d), id: (batch_size, m) => (seq_len, batch_size, k)
        y=nn.Dense(self.r*self.m)(x).reshape(x.shape[0],x.shape[1],self.m,self.r) # (seq_len, batch_size, m, r)
        y=nn.Dense(self.k)(y) # (seq_len, batch_size, m, k)
        y=jnp.sum(y*id[None,:,:,None],axis=-2) # (seq_len, batch_size, k)
        return nn.Dense(self.k)(x) + y

class MLP_3_LoRA(nn.Module):
    # d: int # input dim
    r: int # hidden dim
    k: int # output dim = action dim
    l: int # latent layer dim
    m: int # number of policies

    discrete_policy: bool = False

    @nn.compact
    def __call__(self, x, id): # x: (seq_len, batch_size, d), id: (batch_size, m) => (seq_len, batch_size, k)
        x=jnp.concatenate([x, jnp.broadcast_to(id[None,:,:], (x.shape[0], x.shape[1], id.shape[1]))], axis=-1)
        x=nn.relu(LoRA_mod(self.r, self.l, self.m)(x, id))
        x=nn.relu(LoRA_mod(self.r, self.l, self.m)(x, id))
        if self.discrete_policy:
            x=nn.softmax(LoRA_mod(self.r, self.k, self.m)(x, id))
            pi=distrax.Categorical(logits=x)
        else:
            mu=LoRA_mod(self.r, self.k, self.m)(x, id)
            std=nn.softplus(LoRA_mod(self.r, self.k, self.m)(x, id))+1e-5
            pi=distrax.MultivariateNormalDiag(loc=mu, scale_diag=std)
        return pi

# class EncoderA(nn.Module):
#     latent_dim: int  # Latent space dimension
#     hidden_dim: int  # Hidden state dimension
# 
#     @nn.compact
#     def __call__(self, x):
#         obs, dones = x  # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size)
# 
#         # Embedding layer
#         embedding = nn.Dense(
#             128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(obs)
#         embedding = nn.relu(embedding)
# 
#         # RNN hidden state initialization
#         batch_size = embedding.shape[1]
#         hidden = ScannedRNN.initialize_carry(batch_size, self.hidden_dim)
# 
#         # RNN processing
#         rnn_in = (embedding, dones)
#         hidden, embedding = ScannedRNN()(hidden, rnn_in)
# 
#         # Compute latent space parameters
#         m = nn.Dense(self.latent_dim)(embedding)
#         return m
# class DecoderA(nn.Module):
#     action_dim: int  # action space dimension
# 
#     @nn.compact
#     def __call__(self, z, x):
#         obs, dones = x # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size)
#         
#         embedding = nn.Dense(
#             128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(obs)
#         embedding = nn.relu(embedding)
#         embedding = nn.Dense(
#             32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(embedding)
#         embedding = nn.relu(embedding)
#         embedding = jnp.concatenate([embedding, z], axis=-1)
#         
#         actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
#         
#         pi = distrax.Categorical(logits=actor_logits)
#         return pi
# class ContinuousDecoderA(nn.Module):
#     action_dim: int  # action space dimension
# 
#     @nn.compact
#     def __call__(self, z, x):
#         obs, dones = x # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size)
# 
#         
#         embedding = nn.Dense(
#             128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(obs)
#         embedding = nn.relu(embedding)
#         embedding = nn.Dense(
#             32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
#         )(embedding)
#         embedding = nn.relu(embedding)
#         embedding = jnp.concatenate([embedding, z], axis=-1)
#         actor_std = nn.Dense(
#             self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
#         )(embedding)
#         actor_std = jax.nn.softplus(actor_std) + 1e-5
#         
#         actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
#         
#         pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=actor_std)
#         return pi
# class DEC_allstep(nn.Module):
#     latent_dim: int
#     n_clusters: int
#     action_dim: int
#     discrete_action: bool
# 
#     def setup(self):
#         self.encoder = EncoderA(self.latent_dim, 32)
#         self.cluster_layer = ClusteringLayer(self.n_clusters, self.latent_dim)
#         self.decoder = DecoderA(self.action_dim) if self.discrete_action else ContinuousDecoderA(self.action_dim)
# 
#     # def encode(self, x):
#     #     return self.encoder.encoder(x)[0]
# 
#     # q (seq_len, batch_size, n_clusters)
#     def __call__(self, x, rng):
#         z = self.encoder(x)
#         q = jax.vmap(self.cluster_layer, in_axes=(0,))(z)
#         # q = jnp.sum(jnp.log(q), axis=0)
#         x_hat = self.decoder(z, x)
#         return x_hat, q, z