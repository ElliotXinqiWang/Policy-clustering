from flax import linen as nn
import functools
from flax.linen.initializers import constant, orthogonal
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
class Encoder(nn.Module):
    latent_dim: int  # Latent space dimension
    hidden_dim: int  # Hidden state dimension

    @nn.compact
    def __call__(self, x):
        obs, dones = x  # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size)

        # Embedding layer
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        # RNN hidden state initialization
        batch_size = embedding.shape[1]
        hidden = ScannedRNN.initialize_carry(batch_size, self.hidden_dim)

        # RNN processing
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Index first 'done'
        first_done = jnp.argmax(dones, axis=0)
        first_done = jnp.where(jnp.any(dones, axis=0), first_done, obs.shape[0] - 1)

        # Select embeddings based on first_done
        batch_indices = jnp.arange(batch_size)
        needed_embedding = embedding[first_done, batch_indices]

        # Compute latent space parameters
        mu = nn.Dense(self.latent_dim)(needed_embedding)
        log_var = nn.Dense(self.latent_dim)(needed_embedding)
        return mu, log_var # (batch_size, latent_dim)
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
class DiscretePolicyVAE(nn.Module):
    latent_dim: int
    Encoder_hidden_dim: int
    action_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim, self.Encoder_hidden_dim)
        self.decoder = Decoder(self.action_dim)

    def reparameterize(self, mu, log_var, rng):
        """Reparameterization trick."""
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, mu.shape)
        return mu + eps * std

    def __call__(self, x, rng):
        mu, log_var = self.encoder(x)  # Encode
        z = self.reparameterize(mu, log_var, rng)  # Reparameterization
        pi = self.decoder(z, x)  # Decode
        return pi, mu, log_var
class ContinuousPolicyVAE(nn.Module):
    latent_dim: int
    Encoder_hidden_dim: int
    action_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim, self.Encoder_hidden_dim)
        self.decoder = ContinuousDecoder(self.action_dim)

    def reparameterize(self, mu, log_var, rng):
        """Reparameterization trick."""
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, mu.shape)
        return mu + eps * std

    def __call__(self, x, rng):
        mu, log_var = self.encoder(x)  # Encode
        z = self.reparameterize(mu, log_var, rng)  # Reparameterization
        pi = self.decoder(z, x)  # Decode
        return pi, mu, log_var
class VQVAE(nn.Module):
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

    def reparameterize(self, z):
        euc_dis = jnp.sum((z[:,None,:] - self.codebook[None,:,:])**2, axis=-1)
        z_q = jnp.argmin(euc_dis, axis=-1)
        z_q = self.codebook[z_q]
        loss = jnp.mean(jax.lax.stop_gradient(z_q) - z)**2 * self.alpha + \
               jnp.mean(z - jax.lax.stop_gradient(z_q))**2 * self.beta
        return z + jax.lax.stop_gradient(z_q - z), loss

    def __call__(self, x):
        mu, log_var = self.encoder(x)  # Encode
        z, loss = self.reparameterize(mu)  # Reparameterization
        print(mu.shape,z.shape)
        pi = self.decoder(z, x)  # Decode
        return pi, z, loss
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
        z_e = jax.nn.softmax(euc_dis)
        z_q = jnp.argmax(z_e, axis=-1)
        z_q = self.codebook[z_q]
        loss = jnp.mean(jax.lax.stop_gradient(z_q) - z)**2 * self.alpha + \
               jnp.mean(z - jax.lax.stop_gradient(z_q))**2 * self.beta
        return z + jax.lax.stop_gradient(z_q - z), z_e, loss

    def __call__(self, x, rng, it):
        mu, log_var = self.encoder(x)  # Encode
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
class ContinuousDEC(nn.Module):
    latent_dim: int
    n_clusters: int
    action_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim, 32)
        self.cluster_layer = ClusteringLayer(self.n_clusters, self.latent_dim)
        self.decoder = ContinuousDecoder(self.action_dim)

    def encode(self, x):
        return self.encoder.encoder(x)[0]

    def __call__(self, x, rng):
        z = self.encoder(x)[0]
        q = self.cluster_layer(z)
        x_hat = self.decoder(z, x)
        return x_hat, q, z
class DiscreteDEC(nn.Module):
    latent_dim: int
    n_clusters: int
    action_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim, 32)
        self.cluster_layer = ClusteringLayer(self.n_clusters, self.latent_dim)
        self.decoder = Decoder(self.action_dim)

    def encode(self, x):
        return self.encoder.encoder(x)[0]

    # q (batch_size, n_clusters)
    def __call__(self, x, rng):
        z = self.encoder(x)[0]
        q = self.cluster_layer(z)
        x_hat = self.decoder(z, x)
        return x_hat, q, z
class EncoderA(nn.Module):
    latent_dim: int  # Latent space dimension
    hidden_dim: int  # Hidden state dimension

    @nn.compact
    def __call__(self, x):
        obs, dones = x  # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size)

        # Embedding layer
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        # RNN hidden state initialization
        batch_size = embedding.shape[1]
        hidden = ScannedRNN.initialize_carry(batch_size, self.hidden_dim)

        # RNN processing
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Compute latent space parameters
        m = nn.Dense(self.latent_dim)(embedding)
        return m
class DecoderA(nn.Module):
    action_dim: int  # action space dimension

    @nn.compact
    def __call__(self, z, x):
        obs, dones = x # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size)
        
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        embedding = jnp.concatenate([embedding, z], axis=-1)
        
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        
        pi = distrax.Categorical(logits=actor_logits)
        return pi
class ContinuousDecoderA(nn.Module):
    action_dim: int  # action space dimension

    @nn.compact
    def __call__(self, z, x):
        obs, dones = x # obs: (seq_len, batch_size, obs_dim), dones: (seq_len, batch_size)

        
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
class DiscreteDEC_allstep(nn.Module):
    latent_dim: int
    n_clusters: int
    action_dim: int

    def setup(self):
        self.encoder = EncoderA(self.latent_dim, 32)
        self.cluster_layer = ClusteringLayer(self.n_clusters, self.latent_dim)
        self.decoder = DecoderA(self.action_dim)

    # def encode(self, x):
    #     return self.encoder.encoder(x)[0]

    # q (seq_len, batch_size, n_clusters)
    def __call__(self, x, rng):
        z = self.encoder(x)
        q = jax.vmap(self.cluster_layer, in_axes=(0,))(z)
        # q = jnp.sum(jnp.log(q), axis=0)
        x_hat = self.decoder(z, x)
        return x_hat, q, z
class ContinuousDEC_allstep(nn.Module):
    latent_dim: int
    n_clusters: int
    action_dim: int

    def setup(self):
        self.encoder = EncoderA(self.latent_dim, 32)
        self.cluster_layer = ClusteringLayer(self.n_clusters, self.latent_dim)
        self.decoder = ContinuousDecoderA(self.action_dim)

    # def encode(self, x):
    #     return self.encoder.encoder(x)[0]

    # q (seq_len, batch_size, n_clusters)
    def __call__(self, x, rng):
        z = self.encoder(x)
        q = jax.vmap(self.cluster_layer, in_axes=(0,))(z)
        # q = jnp.sum(jnp.log(q), axis=0)
        x_hat = self.decoder(z, x)
        return x_hat, q, z