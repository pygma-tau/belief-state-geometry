import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import Dict, List, Tuple

class HMM:
    def __init__(self, transition_matrices: Dict[str, np.ndarray], initial_state_dist=None):
        """
        Initialize a Hidden Markov Model.

        Args:
            transition_matrices: Dictionary mapping tokens to transition matrices.
                                 Where T[i,j] = Pr(token, s_j | s_i)
            initial_state_dist: Initial distribution over states. If None, use uniform distribution.
        """
        self.tokens = list(transition_matrices.keys())
        self.transition_matrices = transition_matrices

        # Validate all matrices have same shape
        shapes = [mat.shape for mat in transition_matrices.values()]
        if len(set(shapes)) > 1:
            raise ValueError("All transition matrices must have the same shape")

        self.n_states = shapes[0][0]

        # Set initial state distribution
        if initial_state_dist is None:
            self.initial_state_dist = np.ones(self.n_states) / self.n_states
        else:
            if len(initial_state_dist) != self.n_states:
                raise ValueError("Initial state distribution must match number of states")
            self.initial_state_dist = initial_state_dist / np.sum(initial_state_dist)

        # Compute the stationary distribution
        self.compute_stationary_distribution()

    def compute_stationary_distribution(self):
        """Compute the stationary distribution of the HMM."""
        # Sum the transition matrices to get the overall transition matrix
        total_matrix = sum(self.transition_matrices.values())

        # Find the eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(total_matrix.T)
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])
        self.stationary_dist = stationary / np.sum(stationary)

    def generate_sequence(self, length: int, start_state=None) -> Tuple[List[str], List[int]]:
        """
        Generate a sequence of tokens from the HMM.

        Args:
            length: Length of the sequence to generate
            start_state: Starting state. If None, sample from initial_state_dist

        Returns:
            (token_sequence, state_sequence)
        """
        if start_state is None:
            current_state = np.random.choice(self.n_states, p=self.initial_state_dist)
        else:
            current_state = start_state

        tokens = []
        states = [current_state]

        for _ in range(length):
            # Compute the total probability distribution for transitioning from current state
            # and all possible emissions
            probs = []
            token_choices = []

            for token, matrix in self.transition_matrices.items():
                for next_state in range(self.n_states):
                    prob = matrix[current_state, next_state]
                    if prob > 0:
                        probs.append(prob)
                        token_choices.append((token, next_state))

            # Normalize probabilities
            probs = np.array(probs)
            probs = probs / np.sum(probs)

            # Sample next token and state
            idx = np.random.choice(len(probs), p=probs)
            token, next_state = token_choices[idx]

            tokens.append(token)
            current_state = next_state
            states.append(current_state)

        return tokens, states[:-1]  # Remove the last state as it doesn't emit a token

    def update_belief(self, belief: np.ndarray, token: str) -> np.ndarray:
        """
        Update the belief state given a new observation.

        Args:
            belief: Current belief state (probability distribution over states)
            token: Observed token

        Returns:
            Updated belief state
        """
        if token not in self.transition_matrices:
            raise ValueError(f"Unknown token: {token}")

        T_x = self.transition_matrices[token]

        # Calculate η' = ηT(x) / ηT(x)1 (as per equation 1 in the paper)
        numerator = belief @ T_x
        denominator = np.sum(numerator)

        if denominator == 0:
            # If the token is impossible given the current belief,
            # reset to uniform distribution
            return np.ones(self.n_states) / self.n_states

        return numerator / denominator

    def compute_belief_sequence(self, tokens: List[str], initial_belief=None) -> List[np.ndarray]:
        """
        Compute a sequence of belief states from a sequence of tokens.

        Args:
            tokens: Sequence of tokens
            initial_belief: Initial belief state. If None, use the stationary distribution.

        Returns:
            Sequence of belief states
        """
        if initial_belief is None:
            belief = self.stationary_dist.copy()
        else:
            belief = initial_belief.copy()

        belief_sequence = [belief.copy()]

        for token in tokens:
            belief = self.update_belief(belief, token)
            belief_sequence.append(belief.copy())

        return belief_sequence


# Implement specific HMMs from the paper

def create_z1r_hmm():
    """
    Create the Zero-One-Random (Z1R) process described on page 2.

    This process generates strings of tokens of the form ...01R01R...,
    where R is a randomly generated 0 or 1.
    """
    # States: S0, S1, SR
    n_states = 3

    # Transition matrices
    T_0 = np.zeros((n_states, n_states))
    T_1 = np.zeros((n_states, n_states))

    # S0 --0:100%--> S1
    T_0[0, 1] = 1.0

    # S1 --1:100%--> SR
    T_1[1, 2] = 1.0

    # SR --0:50%--> S0
    # SR --1:50%--> S0
    T_0[2, 0] = 0.5
    T_1[2, 0] = 0.5

    return HMM({'0': T_0, '1': T_1})

def create_mess3_hmm():
    """
    Create the Mess3 process as described in Figure 5.

    This process has 3 hidden states and generates sequences in a token vocabulary of {A, B, C}.
    """
    # Define transition matrices as described in the appendix (page 14)
    T_A = np.array([
        [0.765, 0.00375, 0.00375],
        [0.0425, 0.0675, 0.00375],
        [0.0425, 0.00375, 0.0675]
    ])

    T_B = np.array([
        [0.0675, 0.0425, 0.00375],
        [0.00375, 0.765, 0.00375],
        [0.00375, 0.0425, 0.0675]
    ])

    T_C = np.array([
        [0.0675, 0.00375, 0.0425],
        [0.00375, 0.0675, 0.0425],
        [0.00375, 0.00375, 0.765]
    ])

    return HMM({'A': T_A, 'B': T_B, 'C': T_C})

def create_rrxor_hmm():
    """
    Create the Random-Random-XOR (RRXOR) process described in Figure 7.

    This process has 5 states and a vocabulary of {0, 1}.
    """
    # Define transition matrices as described in the appendix (page 15)
    T_0 = np.array([
        [0, 0.5, 0, 0, 0],
        [0, 0, 0, 0, 0.5],
        [0, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ])

    T_1 = np.array([
        [0, 0, 0.5, 0, 0],
        [0, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 0.5],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    return HMM({'0': T_0, '1': T_1})

def plot_belief_simplex(beliefs, labels=None, figsize=(10, 8)):
    """
    Plot belief states in a probability simplex.

    Args:
        beliefs: List of belief states (each a probability distribution)
        labels: Optional labels for each belief state
        figsize: Figure size
    """
    n_states = beliefs[0].shape[0]

    if n_states == 3:
        # For 3 states, we can plot directly in a 2D triangle
        fig, ax = plt.subplots(figsize=figsize)

        # Define the triangle vertices
        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

        # Draw the triangle
        poly = Polygon(triangle, fill=False, edgecolor='black')
        ax.add_patch(poly)

        # Plot each belief state
        for i, belief in enumerate(beliefs):
            # Convert from barycentric to Cartesian coordinates
            x = belief[0] * triangle[0, 0] + belief[1] * triangle[1, 0] + belief[2] * triangle[2, 0]
            y = belief[0] * triangle[0, 1] + belief[1] * triangle[1, 1] + belief[2] * triangle[2, 1]

            color = (belief[0], belief[1], belief[2])
            ax.scatter(x, y, color=color, s=50)

            if labels is not None and i < len(labels):
                ax.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points')

        # Label the vertices
        ax.annotate("State 0", triangle[0], xytext=(10, -10), textcoords='offset points')
        ax.annotate("State 1", triangle[1], xytext=(-10, -10), textcoords='offset points')
        ax.annotate("State 2", triangle[2], xytext=(0, 10), textcoords='offset points')

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_aspect('equal')
        ax.axis('off')

    else:
        # For more states, we'd need dimensionality reduction or projection
        print(f"Cannot directly visualize simplex with {n_states} states")

    plt.title("Belief State Simplex")
    plt.tight_layout()
    return fig, ax

# Example usage
if __name__ == "__main__":
    # Create an instance of each HMM
    z1r_hmm = create_z1r_hmm()
    mess3_hmm = create_mess3_hmm()
    rrxor_hmm = create_rrxor_hmm()

    # Generate sequences
    seq_length = 30
    z1r_tokens, z1r_states = z1r_hmm.generate_sequence(seq_length)
    mess3_tokens, mess3_states = mess3_hmm.generate_sequence(seq_length)
    rrxor_tokens, rrxor_states = rrxor_hmm.generate_sequence(seq_length)

    # Print some example sequences
    print("Z1R sequence:", "".join(z1r_tokens))
    print("Mess3 sequence:", "".join(mess3_tokens))
    print("RRXOR sequence:", "".join(rrxor_tokens))

    # Compute belief sequences
    z1r_beliefs = z1r_hmm.compute_belief_sequence(z1r_tokens)
    mess3_beliefs = mess3_hmm.compute_belief_sequence(mess3_tokens)

    # Plot belief simplex for the 3-state models
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_belief_simplex(z1r_beliefs)
    plt.title("Z1R Belief State Geometry")

    plt.subplot(1, 2, 2)
    plot_belief_simplex(mess3_beliefs)
    plt.title("Mess3 Belief State Geometry")

    plt.tight_layout()
    plt.show()
