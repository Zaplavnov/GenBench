import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients, DeepLift
from Bio import motifs
from Bio.motifs import jaspar
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from transformers import AutoTokenizer

import src.utils as utils
from src.dataloaders import SequenceDataset
from src.utils import registry
from src.utils.tokenizer_dna import one_hot_encode_dna

def load_model_from_checkpoint(config):
    """
    Load model from checkpoint
    """
    # Instantiate model and load checkpoint
    pl_module = utils.instantiate(registry.model, config.model)
    checkpoint = torch.load(config.ckpt_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    if all(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k[6:]: v for k, v in state_dict.items()}
    
    # Load weights
    pl_module.load_state_dict(state_dict, strict=False)
    pl_module.eval()
    
    return pl_module

def get_tokenizer(config):
    """
    Get tokenizer based on config
    """
    if config.dataset.tokenizer_name == "hyena":
        # Using character-level tokenization for hyena
        return lambda x: x
    else:
        # Using HuggingFace tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_path)
        return tokenizer

def tokenize_sequence(seq, tokenizer, include_n=True):
    """
    Tokenize sequence based on tokenizer type
    
    Args:
        seq: Input DNA sequence string
        tokenizer: Tokenizer to use or string indicating tokenization type
        include_n: Whether to include N as a separate channel in one-hot encoding (default: True)
                  If False, N will be encoded as [0.25, 0.25, 0.25, 0.25]
    
    Returns:
        Tokenized sequence in the appropriate format
    """
    if isinstance(tokenizer, str) or tokenizer is None or callable(tokenizer) and tokenizer.__name__ == "<lambda>":
        # Character-level tokenization (one-hot encoding)
        return one_hot_encode_dna(seq, include_n=include_n)
    else:
        # Using tokenizer from transformers
        return tokenizer(seq, return_tensors="pt")

def compute_attribution(model, inputs, method="ig", target_class=0):
    """
    Compute attribution scores using Captum
    
    Args:
        model: The model to explain
        inputs: Input tensor
        method: Attribution method ('saliency', 'ig', or 'dl')
        target_class: Target class for attribution
        
    Returns:
        Numpy array of attribution scores
    """
    model.eval()
    
    # Define forward function for Captum
    def forward_func(x):
        return model(x)
    
    # Select attribution method
    if method == "saliency":
        explainer = Saliency(forward_func)
        attr = explainer.attribute(inputs, target=target_class)
    elif method == "ig":
        explainer = IntegratedGradients(forward_func)
        baseline = torch.zeros_like(inputs)
        attr = explainer.attribute(inputs, baseline, target=target_class, n_steps=50)
    elif method == "dl":
        explainer = DeepLift(forward_func)
        baseline = torch.zeros_like(inputs)
        attr = explainer.attribute(inputs, baseline, target=target_class)
    else:
        raise ValueError(f"Unknown attribution method: {method}")
    
    # Sum attributions across channels (if any)
    if attr.dim() > 2:
        attr = attr.sum(dim=2)
    
    # Return absolute values and convert to numpy
    return attr.abs().squeeze().detach().cpu().numpy()

def load_jaspar_motif(motif_id):
    """
    Load a motif from JASPAR database by ID
    
    Args:
        motif_id: JASPAR motif ID (e.g., 'MA0108.1')
    
    Returns:
        Position-specific scoring matrix (PSSM)
    """
    try:
        jdb = jaspar.JASPAR5()
        pwm = jdb.fetch_motif_by_id(motif_id)
        pssm = pwm.counts.normalize(pseudocounts=1).log_odds()
        return pssm
    except Exception as e:
        print(f"Error loading JASPAR motif: {e}")
        return None

def correlate_with_motif(seq_scores, motif_pssm, seq):
    """
    Correlate attribution scores with a motif PSSM
    
    Args:
        seq_scores: Attribution scores for sequence
        motif_pssm: Position-specific scoring matrix
        seq: Original sequence
    
    Returns:
        Array of correlation scores
    """
    if motif_pssm is None:
        return None
    
    correlations = []
    L, k = len(seq_scores), motif_pssm.length
    
    # One-hot encode sequence
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    for i in range(L - k + 1):
        window = seq_scores[i:i+k]
        subseq = seq[i:i+k]
        
        # Skip windows with 'N'
        if 'N' in subseq:
            correlations.append(0)
            continue
        
        # Convert subsequence to one-hot
        one_hot = np.zeros((k, 4))
        for j, nuc in enumerate(subseq):
            if nuc in mapping and mapping[nuc] < 4:  # Skip 'N'
                one_hot[j, mapping[nuc]] = 1
        
        # Calculate motif scores for this window
        motif_scores = np.array([
            sum(one_hot[j, b] * motif_pssm[j, b] 
                for b in range(4) if one_hot[j, b] > 0)
            for j in range(k)
        ])
        
        # Calculate correlation
        if np.std(window) > 0 and np.std(motif_scores) > 0:
            corr = np.corrcoef(window, motif_scores)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0)
    
    return np.array(correlations)

def visualize_attribution(seq, attribution, motif_corr=None, save_path=None):
    """
    Visualize attribution scores and optional motif correlation
    
    Args:
        seq: Original sequence
        attribution: Attribution scores
        motif_corr: Optional motif correlation scores
        save_path: Path to save visualization
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot attribution scores
    ax.bar(range(len(attribution)), attribution, alpha=0.7, label="Attribution")
    ax.set_xlabel("Position")
    ax.set_ylabel("Attribution Score")
    
    # Plot motif correlation if available
    if motif_corr is not None:
        # Create second y-axis for correlation
        ax2 = ax.twinx()
        # Pad correlation array to match sequence length
        pad_size = len(attribution) - len(motif_corr)
        padded_corr = np.pad(motif_corr, (0, pad_size), 'constant')
        ax2.plot(range(len(padded_corr)), padded_corr, 'r-', label="Motif Correlation")
        ax2.set_ylabel("Motif Correlation")
        ax2.legend(loc="upper right")
    
    ax.legend(loc="upper left")
    plt.title("Sequence Attribution Analysis")
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def run_interpretation(config):
    """
    Run interpretation pipeline
    
    Args:
        config: Configuration object
    """
    print(f"Loading model from checkpoint: {config.ckpt_path}")
    model = load_model_from_checkpoint(config)
    
    print(f"Loading tokenizer: {config.dataset.tokenizer_name}")
    tokenizer = get_tokenizer(config)
    
    print(f"Loading dataset: {config.dataset.dataset_name}")
    dataset = SequenceDataset.registry[config.dataset._name_](**config.dataset)
    dataset.setup()
    
    # Get dataset samples
    data_loader = dataset.train_dataloader()
    samples = []
    
    print("Collecting samples for interpretation...")
    for i, batch in enumerate(data_loader):
        if i >= config.num_samples:
            break
        
        if isinstance(batch, dict):
            seq = batch['input_ids']
            if isinstance(seq, torch.Tensor):
                seq = seq.numpy().tolist()
                # Convert back to sequence if tokenized
                if config.dataset.tokenizer_name != "hyena":
                    # This is a simplification, might need adjustment based on tokenizer
                    seq = tokenizer.decode(seq)
            samples.append(seq)
        else:
            samples.append(batch)
    
    # Get include_n parameter with default True
    include_n = getattr(config.interpret, "include_n", True)
    print(f"Using one-hot encoding with include_n={include_n}")
    
    print(f"Computing attributions using method: {config.attribution_method}")
    for i, sample in enumerate(tqdm(samples[:config.num_samples])):
        # Tokenize sequence
        inputs = tokenize_sequence(sample, tokenizer, include_n=include_n)
        
        # Compute attribution
        attribution = compute_attribution(
            model, 
            inputs, 
            method=config.attribution_method,
            target_class=config.target_class
        )
        
        # Normalize attribution
        norm_attr = attribution / (np.max(attribution) + 1e-8)
        
        # Correlate with motif if specified
        motif_corr = None
        if config.motif_id:
            print(f"Correlating with JASPAR motif: {config.motif_id}")
            motif_pssm = load_jaspar_motif(config.motif_id)
            if motif_pssm:
                motif_corr = correlate_with_motif(norm_attr, motif_pssm, sample)
        
        # Visualize and save results
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = os.path.join(output_dir, f"attribution_{i}.png")
        visualize_attribution(sample, norm_attr, motif_corr, save_path)
        
        # Save raw attribution data
        np.save(
            os.path.join(output_dir, f"attribution_{i}.npy"), 
            {"sequence": sample, "attribution": norm_attr, "motif_correlation": motif_corr}
        )
    
    print(f"Interpretation complete. Results saved to {config.output_dir}")

@hydra.main(config_path="../../configs", config_name="config.yaml")
def interpret_cli(config: DictConfig):
    # Set defaults if not specified
    if "interpret" not in config:
        config.interpret = {
            "attribution_method": "ig",
            "num_samples": 10,
            "motif_id": None,
            "target_class": 0,
            "output_dir": "interpret_results"
        }
    
    run_interpretation(config)

if __name__ == "__main__":
    interpret_cli() 