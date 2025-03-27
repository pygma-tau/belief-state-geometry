import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List

# Import our modules
from hmm import create_z1r_hmm, create_mess3_hmm, create_rrxor_hmm
from hmm_dataset import HMMDataset, ParameterizableHMM, create_hmm_dataloaders
from hmm_transformer import HMMTransformer, BeliefLoss


def get_device():
    """Get the best available device: MPS (Apple Silicon) > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_transformer(
    vocab_size: int,
    predict_beliefs: bool = False,
    num_states: Optional[int] = None,
    **kwargs
) -> HMMTransformer:
    """Create a transformer model with specified parameters"""
    model_kwargs = {
        'vocab_size': vocab_size,
        'd_model': kwargs.get('d_model', 128),
        'nhead': kwargs.get('nhead', 4),
        'num_layers': kwargs.get('num_layers', 3),
        'dim_feedforward': kwargs.get('dim_feedforward', 512),
        'dropout': kwargs.get('dropout', 0.1),
        'predict_beliefs': predict_beliefs,
    }

    if predict_beliefs:
        model_kwargs['num_states'] = num_states

    return HMMTransformer(**model_kwargs)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    belief_criterion: Optional[nn.Module],
    device: torch.device,
    belief_weight: float = 1.0
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function for token prediction
        belief_criterion: Loss function for belief prediction (if applicable)
        device: Device to train on
        belief_weight: Weight for belief loss term

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    token_loss_total = 0
    belief_loss_total = 0
    correct_preds = 0
    total_preds = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()

        # Get the inputs
        tokens = batch['tokens'].to(device)

        # Shift for prediction: input is all tokens except last, target is all tokens except first
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]

        # Forward pass
        output = model(input_tokens)
        token_logits = output['token_logits']

        # Compute token prediction loss
        token_loss = criterion(
            token_logits.reshape(-1, token_logits.size(-1)),
            target_tokens.reshape(-1)
        )

        # Initialize belief loss
        belief_loss = torch.tensor(0.0, device=device)

        # Compute belief prediction loss if applicable
        if 'belief_probs' in output and 'beliefs' in batch:
            target_beliefs = batch['beliefs'][:, 1:].to(device)  # Shift by 1 like tokens
            belief_loss = belief_criterion(
                output['belief_probs'],
                target_beliefs
            )

            # Track belief loss
            belief_loss_total += belief_loss.item()

        # Combine losses
        loss = token_loss + belief_weight * belief_loss

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        token_loss_total += token_loss.item()

        # Compute accuracy
        _, predicted = torch.max(token_logits, dim=-1)
        correct_preds += (predicted == target_tokens).sum().item()
        total_preds += target_tokens.numel()

    # Compute averages
    avg_loss = total_loss / len(train_loader)
    avg_token_loss = token_loss_total / len(train_loader)
    avg_belief_loss = belief_loss_total / len(train_loader) if belief_loss_total > 0 else 0
    accuracy = correct_preds / total_preds if total_preds > 0 else 0

    return {
        'loss': avg_loss,
        'token_loss': avg_token_loss,
        'belief_loss': avg_belief_loss,
        'accuracy': accuracy
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    belief_criterion: Optional[nn.Module],
    device: torch.device,
    belief_weight: float = 1.0
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function for token prediction
        belief_criterion: Loss function for belief prediction (if applicable)
        device: Device to validate on
        belief_weight: Weight for belief loss term

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0
    token_loss_total = 0
    belief_loss_total = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            # Get the inputs
            tokens = batch['tokens'].to(device)

            # Shift for prediction
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            # Forward pass
            output = model(input_tokens)
            token_logits = output['token_logits']

            # Compute token prediction loss
            token_loss = criterion(
                token_logits.reshape(-1, token_logits.size(-1)),
                target_tokens.reshape(-1)
            )

            # Initialize belief loss
            belief_loss = torch.tensor(0.0, device=device)

            # Compute belief prediction loss if applicable
            if 'belief_probs' in output and 'beliefs' in batch:
                target_beliefs = batch['beliefs'][:, 1:].to(device)  # Shift by 1 like tokens
                belief_loss = belief_criterion(
                    output['belief_probs'],
                    target_beliefs
                )

                # Track belief loss
                belief_loss_total += belief_loss.item()

            # Combine losses
            loss = token_loss + belief_weight * belief_loss

            # Track metrics
            total_loss += loss.item()
            token_loss_total += token_loss.item()

            # Compute accuracy
            _, predicted = torch.max(token_logits, dim=-1)
            correct_preds += (predicted == target_tokens).sum().item()
            total_preds += target_tokens.numel()

    # Compute averages
    avg_loss = total_loss / len(val_loader)
    avg_token_loss = token_loss_total / len(val_loader)
    avg_belief_loss = belief_loss_total / len(val_loader) if belief_loss_total > 0 else 0
    accuracy = correct_preds / total_preds if total_preds > 0 else 0

    return {
        'loss': avg_loss,
        'token_loss': avg_token_loss,
        'belief_loss': avg_belief_loss,
        'accuracy': accuracy
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    belief_criterion: Optional[nn.Module],
    device: torch.device,
    epochs: int = 10,
    belief_weight: float = 1.0,
    patience: int = 5,
    model_save_path: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the model.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        criterion: Loss function for token prediction
        belief_criterion: Loss function for belief prediction (if applicable)
        device: Device to train on
        epochs: Number of epochs to train for
        belief_weight: Weight for belief loss term
        patience: Early stopping patience
        model_save_path: Path to save the best model

    Returns:
        Tuple of (trained model, training history)
    """
    # Initialize tracking variables
    history = {
        'train_loss': [],
        'train_token_loss': [],
        'train_belief_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_token_loss': [],
        'val_belief_loss': [],
        'val_accuracy': []
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, belief_criterion, device, belief_weight
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, belief_criterion, device, belief_weight
        )

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_token_loss'].append(train_metrics['token_loss'])
        history['train_belief_loss'].append(train_metrics['belief_loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])

        history['val_loss'].append(val_metrics['loss'])
        history['val_token_loss'].append(val_metrics['token_loss'])
        history['val_belief_loss'].append(val_metrics['belief_loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, Token Loss: {train_metrics['token_loss']:.4f}, "
              f"Belief Loss: {train_metrics['belief_loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Token Loss: {val_metrics['token_loss']:.4f}, "
              f"Belief Loss: {val_metrics['belief_loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        # Check for improvement
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()

            # Save the model if a path was provided
            if model_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'train_loss': train_metrics['loss'],
                }, model_save_path)
                print(f"Model saved to {model_save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def plot_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training history"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot losses
    axs[0, 0].plot(history['train_loss'], label='Train Loss')
    axs[0, 0].plot(history['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Plot token losses
    axs[0, 1].plot(history['train_token_loss'], label='Train Token Loss')
    axs[0, 1].plot(history['val_token_loss'], label='Val Token Loss')
    axs[0, 1].set_title('Token Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    # Plot belief losses if available
    if max(history['train_belief_loss']) > 0:
        axs[1, 0].plot(history['train_belief_loss'], label='Train Belief Loss')
        axs[1, 0].plot(history['val_belief_loss'], label='Val Belief Loss')
        axs[1, 0].set_title('Belief Loss')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()

    # Plot accuracy
    axs[1, 1].plot(history['train_accuracy'], label='Train Accuracy')
    axs[1, 1].plot(history['val_accuracy'], label='Val Accuracy')
    axs[1, 1].set_title('Accuracy')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")

    plt.show()


def test_model_generation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_examples: int = 3,
    context_length: int = 10,
    generation_length: int = 20,
    idx_to_token: Dict[int, str] = None
):
    """
    Test model generation capabilities.

    Args:
        model: The model to test
        test_loader: DataLoader for test data
        device: Device to test on
        num_examples: Number of examples to generate
        context_length: Length of context to use for generation
        generation_length: Length of generated sequence
        idx_to_token: Mapping from indices to tokens
    """
    model.eval()

    # Get a batch from the test loader
    batch = next(iter(test_loader))
    tokens = batch['tokens'].to(device)

    print("\n===== Model Generation Examples =====")

    for i in range(min(num_examples, tokens.size(0))):
        # Get the context
        context = tokens[i, :context_length].unsqueeze(0)

        # Generate from the context
        generated, beliefs = model.generate(context, context_length + generation_length)
        generated = generated[0]  # Remove batch dimension

        # Convert to tokens
        context_tokens = [idx_to_token[idx.item()] for idx in context[0]]
        generated_tokens = [idx_to_token[idx.item()] for idx in generated[context_length:]]

        print(f"\nExample {i+1}:")
        print(f"Context: {''.join(context_tokens)}")
        print(f"Generated: {''.join(generated_tokens)}")

        # Print true continuation if available
        if tokens.size(1) >= context_length + generation_length:
            true_tokens = [idx_to_token[idx.item()] for idx in tokens[i, context_length:context_length+generation_length]]
            print(f"True: {''.join(true_tokens)}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train a transformer model on HMM data')
    parser.add_argument('--hmm_type', default='mess3', choices=['z1r', 'mess3', 'rrxor', 'custom'],
                        help='Type of HMM to use')
    parser.add_argument('--predict_beliefs', action='store_true',
                        help='Whether to predict belief states')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='Length of sequences to generate')
    parser.add_argument('--train_size', type=int, default=8000,
                        help='Number of training sequences')
    parser.add_argument('--val_size', type=int, default=1000,
                        help='Number of validation sequences')
    parser.add_argument('--test_size', type=int, default=1000,
                        help='Number of test sequences')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimensionality of the model')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--belief_weight', type=float, default=1.0,
                        help='Weight for belief loss term')
    parser.add_argument('--patience', type=int, default=500,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', default='output',
                        help='Directory for output files')

    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Determine device
    device = get_device()
    print(f"Using device: {device}")

    # Create HMM and dataloaders
    custom_hmm_params = None
    if args.hmm_type == 'custom':
        custom_hmm_params = {
            'n_states': 5,
            'vocab_size': 4,
            'sparsity': 0.6,
            'concentration': 0.5
        }

    train_loader, val_loader, test_loader, hmm = create_hmm_dataloaders(
        hmm_type=args.hmm_type,
        custom_hmm_params=custom_hmm_params,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        include_beliefs=args.predict_beliefs,
        include_states=False,  # We don't need the true states for training
        seed=args.seed
    )

    # Get vocabulary info
    vocab_size = len(train_loader.dataset.token_to_idx)
    idx_to_token = train_loader.dataset.idx_to_token

    # Create model
    model = create_transformer(
        vocab_size=vocab_size,
        predict_beliefs=args.predict_beliefs,
        num_states=hmm.n_states if args.predict_beliefs else None,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # Define loss functions
    token_criterion = nn.CrossEntropyLoss()
    belief_criterion = BeliefLoss() if args.predict_beliefs else None

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Define model save path
    model_save_path = os.path.join(args.output_dir, f"{args.hmm_type}_transformer.pt")

    # Train the model
    start_time = time.time()
    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=token_criterion,
        belief_criterion=belief_criterion,
        device=device,
        epochs=args.epochs,
        belief_weight=args.belief_weight,
        patience=args.patience,
        model_save_path=model_save_path
    )
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Plot training history
    history_plot_path = os.path.join(args.output_dir, f"{args.hmm_type}_training_history.png")
    plot_history(history, save_path=history_plot_path)

    # Test model generation
    test_model_generation(
        model=model,
        test_loader=test_loader,
        device=device,
        num_examples=3,
        context_length=10,
        generation_length=20,
        idx_to_token=idx_to_token
    )


if __name__ == "__main__":
    main()
