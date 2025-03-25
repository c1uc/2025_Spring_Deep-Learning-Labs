import torch
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


# TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs["VQ_Configs"])

        self.num_image_tokens = configs["num_image_tokens"]
        self.mask_token_id = configs["num_codebook_vectors"]
        self.choice_temperature = configs["choice_temperature"]
        self.gamma = self.gamma_func(configs["gamma_type"])
        self.transformer = BidirectionalTransformer(configs["Transformer_param"])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs["VQ_config_path"], "r"))
        model = VQGAN(cfg["model_param"])
        model.load_state_dict(torch.load(configs["VQ_CKPT_path"]), strict=True)
        model = model.eval()
        return model

    ##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        z_q, z_idx, _ = self.vqgan.encode(x)
        return z_q, z_idx.reshape(z_q.shape[0], -1)

    ##TODO2 step1-2:
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1].
        During training, the input ratio is uniformly sampled;
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.

        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda x: 1 - x
        elif mode == "cosine":
            return lambda x: math.cos(x * math.pi / 2)
        elif mode == "square":
            return lambda x: 1 - x ** 2
        else:
            raise NotImplementedError

    ##TODO2 step1-3:
    def forward(self, x):
        z_q, z_idx = self.encode_to_z(x)

        mask = torch.distributions.Bernoulli(probs=0.5 * torch.ones_like(z_idx)).sample().bool()
        masked_idx = z_idx.clone()
        masked_idx[mask] = self.mask_token_id

        logits = self.transformer(masked_idx)

        return logits, z_idx

    ##TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, mask_num, ratio):
        masked_z_idx = z_indices.clone()
        masked_z_idx[mask_bc] = self.mask_token_id
        logits = self.transformer(masked_z_idx)
        # Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.softmax(logits, dim=-1)

        # FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)
        z_indices_predict[~mask_bc] = z_indices[~mask_bc]

        ratio = self.gamma(ratio)
        # predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.Gumbel(0, 1).sample(z_indices_predict.shape).to(z_indices_predict.device)  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)

        z_indices_predict_prob = torch.where(mask_bc, z_indices_predict_prob, torch.tensor(float('inf')))
        confidence = z_indices_predict_prob + temperature * g

        # hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        # sort the confidence for the rank
        sorted_confidence, sorted_indices = torch.sort(confidence, dim=-1)
        # define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values
        mask_bc = torch.zeros_like(mask_bc)
        mask_bc[:, sorted_indices[:, :int(mask_num * ratio)]] = 1
        mask_bc = mask_bc.bool()

        return z_indices_predict, mask_bc


__MODEL_TYPE__ = {"MaskGit": MaskGit}
