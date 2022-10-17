import clip
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from utils.utils import maybe_cuda, AverageMeter

class ClipImageEncoder(torch.nn.Module):
    def __init__(self, n_classes, classes):
        super(ClipImageEncoder, self).__init__()
        self.n_classes = n_classes

        model, _ = clip.load('RN50', 'cpu', jit=False)
        model = model.float()
        self.encoder = model.visual
        self._build_classifier_from_texts(model, classes)
        self.logit_scale = nn.Parameter(model.logit_scale)

    def _build_classifier_from_texts(self, clip_model, classes, prompt_eng=False):
        templates = text_templates if prompt_eng else ['a photo of a {}']
        with torch.no_grad():
            txt_feats = []
            for classname in tqdm(classes):
                classes = [template.format(classname) for template in templates]
                classes = clip.tokenize(classes) # tokenize
                class_embeddings = clip_model.encode_text(classes) # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                txt_feats.append(class_embedding)
            txt_feats = torch.stack(txt_feats, dim=1)

        self.linear = nn.Linear(1024, self.n_classes, bias=False)
        self.linear.weight = torch.nn.Parameter(txt_feats.T)

    def logits(self, x):
        if self.logit_scale > 4.605:
            self.logit_scale.data = torch.tensor(4.605).to(self.logit_scale.device)
        out = self.linear(x)
        out = self.logit_scale.exp() * out
        return out

    def features(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        out = F.normalize(out, dim=1)
        return out

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)
        return logits

if __name__ == '__main__':
    model = ClipImageEncoder(100)
