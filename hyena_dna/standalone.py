import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# Путь к репозиторию hyena-dna
repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyena-dna")
sys.path.append(repo_path)

# Необходимые импорты из репозитория hyena-dna
from src.models.sequence.model import SequenceModel
from src.models.sequence.base import SequenceModule


class StandaloneHyenaDNA:
    """Standalone HyenaDNA model for inference"""
    
    def __init__(self, weights_path, config_path=None, device="cuda"):
        """
        Initialize a standalone HyenaDNA model
        
        Args:
            weights_path: Path to weights checkpoint
            config_path: Path to config JSON file
            device: Device to run inference on
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Infer config path if not provided
        if config_path is None:
            config_path = os.path.join(os.path.dirname(weights_path), "config.json")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        print(f"Loaded model config from: {config_path}")
        
        # Create model
        self.model = self._create_model()
        
        # Load weights
        self._load_weights(weights_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _create_model(self):
        """Create HyenaDNA model from config"""
        model_config = {
            "d_model": self.config["d_model"],
            "n_layer": self.config["n_layer"],
            "d_inner": self.config["d_inner"],
            "vocab_size": self.config["vocab_size"],
            "resid_dropout": self.config["resid_dropout"],
            "embed_dropout": self.config["embed_dropout"],
            "layer": self.config["layer"]
        }
        
        model = SequenceModel(model_config)
        model = model.to(self.device)
        
        return model
    
    def _load_weights(self, weights_path):
        """Load weights from checkpoint"""
        print(f"Loading weights from: {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location="cpu")
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            
            # Remove "model." prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(checkpoint)
    
    def get_embeddings(self, sequences, batch_size=16, layer_idx=-1):
        """
        Extract embeddings from sequences
        
        Args:
            sequences: List of DNA sequences
            batch_size: Batch size for processing
            layer_idx: Layer to extract embeddings from (-1 for last layer)
        
        Returns:
            numpy.ndarray: Embeddings with shape [n_sequences, embedding_dim]
        """
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
                batch = sequences[i:i+batch_size]
                
                # Format input
                inputs = {"sequence": batch}
                
                # Run inference
                outputs = self.model(inputs, return_all_hiddens=True)
                
                # Get hidden states from specified layer
                hidden_states = outputs["hidden_states"]
                layer_output = hidden_states[layer_idx]
                
                # Average over sequence length
                batch_embeddings = layer_output.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


# Helper function to load a standalone HyenaDNA model
def load_hyenadna(weights_path, config_path=None, device="cuda"):
    """Load a standalone HyenaDNA model"""
    model = StandaloneHyenaDNA(weights_path, config_path, device)
    return model 