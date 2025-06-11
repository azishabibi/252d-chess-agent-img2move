from typing import Tuple

import pytorch_lightning as pl
import torch
import timm 
from torch import nn
from torch.nn import functional as F

MAX_PIECES = {
    0: 8,  # White Pawn
    1: 2,  # White Rook
    2: 2,  # White Knight
    3: 2,  # White Bishop
    4: 2,  # White Queen (allowing for one promotion, adjust if more are common)
    5: 1,  # White King
    6: 8,  # Black Pawn
    7: 2,  # Black Rook
    8: 2,  # Black Knight
    9: 2,  # Black Bishop
    10: 2, # Black Queen
    11: 1, # Black King
    # 12: Empty - no count needed
}

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ChessVisionTransformer(pl.LightningModule):
    def __init__(self,
                 model_name: str = 'vit_small_patch14_dinov2.lvd142m',
                 lr: float = 1e-4,
                 lr_backbone: float = 1e-5,
                 weight_decay: float = 1e-4,
                 piece_penalty_weight: float = 0.1):
        super().__init__()

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.focal_loss = FocalLoss()
        self.piece_penalty_weight = piece_penalty_weight
        self.manhattan_loss_weight = None

        self.backbone = timm.create_model(model_name, pretrained=True)
        num_filters = self.backbone.num_features

        self.backbone.head = nn.Identity()

        self.num_squares = 64
        self.num_classes_per_square = 13
        num_target_outputs = self.num_squares * self.num_classes_per_square

        self.classifier = nn.Sequential(
            nn.LayerNorm(num_filters),
            nn.Linear(num_filters, num_filters // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(num_filters // 2, num_target_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits_flat = self.classifier(features)
        logits_per_square = logits_flat.view(-1, self.num_squares, self.num_classes_per_square)
        return logits_per_square

    def calculate_piece_count_penalty(self, preds_cat: torch.Tensor) -> torch.Tensor:
        penalty = 0.0
        for i in range(preds_cat.shape[0]):
            board_preds = preds_cat[i]
            piece_counts = torch.bincount(board_preds, minlength=self.num_classes_per_square)

            for piece_idx, max_count in MAX_PIECES.items():
                if piece_idx < len(piece_counts):
                    count = piece_counts[piece_idx]
                    if count > max_count:
                        penalty += (count - max_count)

        return penalty / preds_cat.shape[0]
    
    def recognition_accuracy(
        y: torch.Tensor,
        preds: torch.Tensor,
        tolerance: int = 0) -> torch.Tensor:
        correct = ((preds == y).sum(axis=1) > 63-tolerance).sum()

        return correct / preds.shape[0]

    def common_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            return_accuracy_and_penalty: bool = False
        ) -> torch.Tensor:

        x, y_one_hot_flat = batch

        logits_per_square = self.forward(x)

        y_one_hot_per_square = y_one_hot_flat.view(-1, self.num_squares, self.num_classes_per_square)
        y_labels_per_square = torch.argmax(y_one_hot_per_square, dim=2)

        loss = self.focal_loss(
            logits_per_square.reshape(-1, self.num_classes_per_square),
            y_labels_per_square.reshape(-1)
        )
        
        results = {"loss": loss}

        if return_accuracy_and_penalty:
            preds_cat_per_square = torch.argmax(logits_per_square, dim=2)
            square_acc = (preds_cat_per_square == y_labels_per_square).float().mean()
            results["square_acc"] = square_acc
            board_acc = self.recognition_accuracy(y_labels_per_square, preds_cat_per_square)
            results["board_acc"] = board_acc

            if self.piece_penalty_weight > 0:
                piece_penalty = self.calculate_piece_count_penalty(preds_cat_per_square)
                results["piece_penalty"] = piece_penalty
                probs_per_square = torch.softmax(logits_per_square, dim=2)
                
                soft_penalty = 0.0
                total_prob_per_piece = torch.sum(probs_per_square, dim=1)

                for piece_idx, max_count in MAX_PIECES.items():
                    if piece_idx < self.num_classes_per_square:
                        current_soft_count = total_prob_per_piece[:, piece_idx]
                        excess = F.relu(current_soft_count - max_count)
                        soft_penalty += torch.sum(excess)
                
                soft_penalty = soft_penalty / x.size(0)
                results["soft_piece_penalty"] = soft_penalty
                results["loss"] += self.piece_penalty_weight * soft_penalty

        return results