import logging

from PIL import Image
import torch
from torch import nn
import numpy as np
import clip

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ClipSearch:
    def __init__(
        self,
        model_name="RN50",
        device=DEVICE,
    ):
        assert model_name in clip.available_models()
        self.device = device
        self.model, self.img_preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_images(self, rgbs):
        """
        rgbs: list of numpy array (in HWC, RGB) or PIL Image
        """
        # returns normalised features
        imgs = []
        for rgb in rgbs:
            if isinstance(rgb, np.ndarray):
                rgb = Image.fromarray(rgb)
            imgs.append(self.img_preprocess(rgb).to(self.device))
        batch = torch.stack(imgs, 0)
        feats = self.model.encode_image(batch)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_texts(self, texts):
        # returns normalised features
        token_texts = clip.tokenize(texts).to(self.device)
        feats = self.model.encode_text(token_texts)
        return feats / feats.norm(dim=-1, keepdim=True)

    def query_with_feats(self, img_feats, text_feats):
        """
        Either img_feats or text_feats must be singular (aka not a batch). That will be treated as the query.
        """
        query_is_text = len(text_feats.shape) == 1
        if query_is_text:
            text_feats = text_feats.unsqueeze(0)
        else:
            assert len(img_feats.shape) == 1
            img_feats = img_feats.unsqueeze(0)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feats @ text_feats.t()  # logits_per_image
        if query_is_text:
            logits = logits.t()  # logits_per_text

        probs = logits.softmax(dim=-1).detach().numpy().flatten()

        return probs
