import torch
import torch.nn as nn
import numpy as np
from gnn import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict
from utils.parser import args

def get_simulator(nnode_in: int, device: str, nmessage_passing_steps = 10, nmlp_layers = 2, connectivity_radius = np.inf):
    nnode_in = 3*(nnode_in - 1)
    simulator = LearnedSimulator(
            nnode_in = nnode_in,  #time_steps * 3 (dimension of velocity)
            particle_dimensions = 3,
            nedge_in = 4, #relative displacement between 2 particles and distances between them
            latent_dim = 128 ,
            nmessage_passing_steps = nmessage_passing_steps,
            nmlp_layers = nmlp_layers ,
            mlp_hidden_dim = 128,
            connectivity_radius = connectivity_radius,
            device = device)

    return simulator


class LearnedSimulator(nn.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
          self,
          particle_dimensions: int,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          device="cpu"):
    """Initializes the model.

    Args:
      particle_dimensions: Dimensionality of the problem.
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      connectivity_radius: Scalar with the radius of connectivity.
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      device: Runtime device (cuda or cpu).

    """
    super(LearnedSimulator, self).__init__()

    self._connectivity_radius = connectivity_radius

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=particle_dimensions,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)

    self._device = device

  def forward(self):
    """Forward hook runs on class instantiation"""
    pass

  def _compute_graph_connectivity(
          self,
          node_features: torch.tensor,
          radius: float,
          batch_size: int,
          add_self_edges: bool = True):
    """Generate graph edges to all particles within a threshold radius
    Args:
      node_features: Node features with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      radius: Threshold to construct edges to all particles within the radius.
      add_self_edges: Boolean flag to include self edge (default: True)
    """

    # Specify examples id for particles
    batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate([22]*batch_size)]).to(self._device)

    # radius_graph accepts r < radius not r <= radius
    # A torch tensor list of source and target nodes with shape (2, nedges)
    edge_index = radius_graph(
        node_features, r=radius, batch=batch_ids, loop=add_self_edges)

    # The flow direction when using in combination with message passing is
    # "source_to_target"
    receivers = edge_index[0, :]
    senders = edge_index[1, :]

    return receivers, senders

  def _encoder_preprocessor(self, position_sequence: torch.tensor):
    """Extracts important features from the position sequence. Returns a tuple
    of node_features (nparticles, 30), edge_index (nparticles, nparticles), and
    edge_features (nparticles, 3).

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
    """
    nparticles = position_sequence.shape[0]
    batch_size = nparticles//22
    most_recent_position = position_sequence[:, -1]  # (n_nodes, 2)
    velocity_sequence = time_diff(position_sequence)

    # Get connectivity of the graph with shape of (nparticles, 2)
    senders, receivers = self._compute_graph_connectivity(most_recent_position, self._connectivity_radius, batch_size)
    node_features = []

    flat_velocity_sequence = velocity_sequence.contiguous().view(nparticles, -1)
    # There are 5 previous steps, with dim 2
    # node_features shape (nparticles, 5 * 2 = 10)
    node_features.append(flat_velocity_sequence)

    # Collect edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius
    # with shape (nedges, 2)
    # normalized_relative_displacements = (
    #     torch.gather(most_recent_position, 0, senders) -
    #     torch.gather(most_recent_position, 0, receivers)
    # ) / self._connectivity_radius
    relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    )

    # Add relative displacement between two particles as an edge feature
    # with shape (nparticles, ndim)
    edge_features.append(relative_displacements)

    # Add relative distance between 2 particles; shape: (nparticles, 1)
    # Edge features has a final shape of (nparticles, ndim + 1 = 3) 
    relative_distances = torch.norm(relative_displacements, dim=-1, keepdim=True)

    edge_features.append(relative_distances)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))

  def _decoder_postprocessor(self, velocities: torch.tensor, position_sequence: torch.tensor):
    """ Compute new position based on acceleration and current position.
    The model produces the output in normalized space so we apply inverse
    normalization.

    Args:
      normalized_acceleration: Normalized acceleration (nparticles, dim).
      position_sequence: Position sequence of shape (nparticles, dim).

    Returns:
      torch.tensor: New position of the particles.

    """
    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]

    new_position = most_recent_position + velocities  # * dt = 1
    return new_position

  def predict_positions(self, current_positions: torch.tensor):
        
    """Predict position based on velocities.

    Args:
      current_positions: Current particle positions (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      next_positions (torch.tensor): Next position of particles.
    """
    node_features, edge_index, edge_features = self._encoder_preprocessor(current_positions)

    predicted_velocities = self._encode_process_decode(node_features, edge_index, edge_features)

    next_positions = self._decoder_postprocessor(predicted_velocities, current_positions)

    return next_positions

  def predict_velocities(
          self,
          next_positions: torch.tensor,
          position_sequence: torch.tensor):
    """
    Produces normalized and predicted acceleration targets.

    Args:
      next_positions: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      Tensors of shape (nparticles_in_batch, dim) with the predicted and target
        normalized accelerations.

    """
    # Perform the forward pass with the position sequence.
    node_features, edge_index, edge_features = self._encoder_preprocessor(position_sequence)
    
    predicted_velocities = self._encode_process_decode(node_features, edge_index, edge_features)

    target_velocities = self._inverse_decoder_postprocessor(next_positions, position_sequence)

    return predicted_velocities, target_velocities


  def _inverse_decoder_postprocessor(
          self,
          next_position: torch.tensor,
          position_sequence: torch.tensor):
    """Inverse of `_decoder_postprocessor`.

      Args:
        next_position: Tensor of shape (nparticles_in_batch, dim) with the
          positions the model should output given the inputs.
        position_sequence: A sequence of particle positions. Shape is
          (nparticles, 6, dim). Includes current + last 5 positions.

      Returns:
        normalized_acceleration (torch.tensor): Normalized acceleration.

    """
    #x_next = x_7, x_previous = x_6
    #we obtain v_next = x_next - x_previous, v_next = v_6

    previous_position = position_sequence[:, -1]
    next_velocity = next_position - previous_position

    return next_velocity

  def save(
          self,
          path: str = 'model.pt'):
    """Save model state

    Args:
      path: Model path
    """
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    """Load model state from file

    Args:
      path: Model path
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
  """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
  return position_sequence[:, 1:] - position_sequence[:, :-1]
