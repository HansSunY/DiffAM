import torch
import torchvision.transforms as transforms
import numpy as np
import clip
from PIL import Image
from utils.text_templates import imagenet_templates


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

class CLIPLoss(torch.nn.Module):
    def __init__(self, device, lambda_makeup_direction , lambda_direction , direction_loss_type='cosine', clip_model='ViT-B/32'):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.lambda_makeup_direction = lambda_makeup_direction
        self.lambda_direction = lambda_direction
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.target_direction      = None
        
        self.direction_loss = DirectionLoss(direction_loss_type)

        self.src_text_features = None
        self.target_text_features = None


    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)
    
    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]
            
    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class)

        src_encoding    = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)
        return self.direction_loss(edit_direction, self.target_direction).mean()

    def clip_makeup_directional_loss(self, src_img: torch.Tensor, non_makeup_img: torch.Tensor, output_img: torch.Tensor, makeup_img: torch.Tensor) -> torch.Tensor:
        non_makeup_encoding = self.get_image_features(non_makeup_img)
        makeup_encoding = self.get_image_features(makeup_img)
        src_encoding    = self.get_image_features(src_img)
        output_encoding = self.get_image_features(output_img)
        self.target_direction = (makeup_encoding - non_makeup_encoding)
        self.target_direction /= (self.target_direction.clone().norm(dim=-1,keepdim=True) + 1e-7)
        edit_direction = (output_encoding - src_encoding)
        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)
        return self.direction_loss(edit_direction, self.target_direction).mean()

    def forward(self, src_img: torch.Tensor, non_makeup_img: torch.Tensor, output_img: torch.Tensor, makeup_img: torch.Tensor, src_txt = None, trg_txt = None):
        clip_loss = 0.0
        if self.lambda_makeup_direction == 1:
            clip_loss += self.clip_makeup_directional_loss(src_img, non_makeup_img, output_img, makeup_img)
        elif self.lambda_direction == 1:
            clip_loss += self.clip_directional_loss(src_img, src_txt, output_img, trg_txt)
        return clip_loss
