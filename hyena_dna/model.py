import os
import sys
import json
import torch
import numpy as np

# Получаем путь к корневой директории проекта
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hyena_dna_path = os.path.join(repo_path, "hyena-dna")

# Добавляем путь к репозиторию в sys.path
sys.path.append(hyena_dna_path)

# Импортируем необходимые модули
from src.models.sequence.model import SequenceModel


class HyenaDNAModel:
    """Wrapper class for HyenaDNA model"""
    
    def __init__(self, config_path=None, weights_path=None, device="cuda"):
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None
        
        if config_path and weights_path:
            self.load_model(config_path, weights_path)
    
    def load_model(self, config_path, weights_path):
        """Load HyenaDNA model from config and weights"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model_config = {
            "d_model": config["d_model"],
            "n_layer": config["n_layer"],
            "d_inner": config["d_inner"],
            "vocab_size": config["vocab_size"],
            "resid_dropout": config["resid_dropout"],
            "embed_dropout": config["embed_dropout"],
            "layer": config["layer"]
        }
        
        # Create model
        self.model = SequenceModel(model_config)
        
        # Load weights
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
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def get_embeddings(self, sequences, batch_size=32, layer_idx=-1):
        """
        Extract embeddings from HyenaDNA model
        
        Args:
            sequences: List of DNA sequences
            batch_size: Batch size for processing
            layer_idx: Layer index to extract embeddings from (-1 for last layer)
            
        Returns:
            numpy.ndarray: Embeddings of shape [n_sequences, embedding_dim]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]
                
                # Format input for HyenaDNA model
                inputs = {"sequence": batch_seqs}
                
                # Forward pass
                outputs = self.model(inputs, return_all_hiddens=True)
                
                # Get hidden states
                hidden_states = outputs["hidden_states"]
                
                # Select layer
                embeddings = hidden_states[layer_idx]
                
                # Average pooling over sequence length
                seq_embeddings = embeddings.mean(dim=1).cpu().numpy()
                all_embeddings.append(seq_embeddings)
        
        return np.vstack(all_embeddings)


# Function to load HyenaDNA model directly
def load_hyenadna_model(config_path, weights_path, device="cuda"):
    """Helper function to load HyenaDNA model"""
    model = HyenaDNAModel(config_path, weights_path, device)
    return model 