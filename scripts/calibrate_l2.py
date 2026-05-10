import os
import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np

class VAEProfiler(nn.Module):
    def __init__(self, input_dim=384, latent_dim=32):
        super(VAEProfiler, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def calibrate_threshold(vae_model, benign_embeddings, target_fpr=0.01, n_samples=50, noise_std=0.25):
    vae_model.eval()
    mses = []
    with torch.no_grad():
        for emb in benign_embeddings:
            emb = emb.unsqueeze(0)
            noisy_embs = emb.repeat(n_samples, 1) + torch.randn_like(emb.repeat(n_samples, 1)) * noise_std
            reconstructions, _, _ = vae_model(noisy_embs)
            mse = torch.mean((noisy_embs - reconstructions) ** 2, dim=1).mean().item()
            mses.append(mse)
    
    mses = np.array(mses)
    threshold = np.quantile(mses, 1.0 - target_fpr)
    return float(threshold)

def main():
    print("Initializing...")
    device = torch.device("cpu") # Force CPU to avoid CUDA errors in map_location
    print(f"Device: {device}")
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    vae_model = VAEProfiler(input_dim=384).to(device)
    
    model_path = "kingsguard_l2_vae.pth"
    print(f"Loading weights from {model_path}...")
    vae_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae_model.eval()
    
    user_cases_path = os.path.join("injecagent_data", "user_cases.jsonl")
    texts = []
    print(f"Reading benign data from {user_cases_path}...")
    with open(user_cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if "User Instruction" in data:
                    texts.append(data["User Instruction"])
                    
    print(f"Extracted {len(texts)} benign samples. Generating embeddings...")
    embeddings = embed_model.encode(texts, convert_to_tensor=True).to(device)
    
    print("Calibrating VAE threshold...")
    threshold = calibrate_threshold(vae_model, embeddings, target_fpr=0.01)
    
    print(f"Calibration complete! Calculated Threshold (theta_VAE): {threshold:.4f}")
    
    out_path = "calibrated_threshold.json"
    with open(out_path, "w") as f:
        json.dump({"theta_VAE": threshold}, f, indent=4)
        
    print(f"Saved threshold to {out_path}")

if __name__ == "__main__":
    main()
