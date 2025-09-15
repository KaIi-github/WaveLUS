import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import logging
import seaborn as sns
import matplotlib as mpl
import torch.nn.functional as F
import torch.distributions as dist

from tqdm import tqdm
from PIL import Image
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, TensorDataset, SubsetRandomSampler, ConcatDataset, Subset
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.utils import resample
from models.WaveLUS import WaveLUS
from utils.metrics import calculate_metrics, print_metrics, plot_roc_auc

seed = 126
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def calculate_specificity(y_true, y_pred, num_classes):
    """
    Calculate the specificity each class

    Parameters:
        y_true (list or np.array): ground truth
        y_pred (list or np.array): prediction
        num_classes (int): number of classes.

    Return:
        specificities (list): Specificity for each class
    """
    specificities = []
    for class_idx in range(num_classes):
        y_true_binary = np.array(y_true) == class_idx
        y_pred_binary = np.array(y_pred) == class_idx

        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        specificity = tn / (tn + fp)
        specificities.append(specificity)
    return specificities


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss

        Parameters:
        - alpha: Class weights, shape [num_classes], default None。
        - gamma: FocalLoss regulatory factor, default 2。
        - reduction: The computation method for losses, supportive 'mean'、'sum' or 'none', default 'mean'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Parameters:
        - inputs: shape [batch_size, num_classes]，no softmax。
        - targets: shape [batch_size], Each element as an index for the class(0 到 num_classes-1)。

        Return:
        - loss: Focal Loss。
        """
        if targets.dim() == 2:
            targets = torch.argmax(targets, dim=1)
        targets = targets.long()

        if targets.dim() != 1:
            raise ValueError(f"Targets should be (1D) or One-Hot (2D), But the current dimension is {targets.dim()}")
        if targets.size(0) != inputs.size(0):
            raise ValueError("Batch size Mismatch")

        # cross_entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Focal Loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Step 1: Function for random reading video frames.
def extract_and_process_frames(video_path, frame_count=64, target_size=(224, 224)):
    """
    Randomly read 'frame_count' frames from a video and process them.

    Parameters:
    - video_path: str, path to the video file.
    - frame_count: int, number of frames to extract.
    - target_size: tuple, resolution of each frame (width, height).

    Returns:
    - frames: list of numpy arrays, processed frames with shape [224, 224, 3].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Can't open the video file!")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Can't get the total number of frames in the video.")
        cap.release()
        return []
    
    actual_frame_count = min(frame_count, total_frames)
    selected_indices = sorted(random.sample(range(total_frames), actual_frame_count))
    
    frames = []
    current_index = 0
    selected_ptr = 0

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx == selected_indices[selected_ptr]:
            frame_resized = cv2.resize(frame, target_size)
            
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            selected_ptr += 1
            if selected_ptr >= actual_frame_count:
                break
    
    cap.release()
    return frames

class VideoDataset(Dataset):
    def __init__(self, data_dir, train_transform=None, val_transform=None, num_frames=64, mode='val'):
        """
        Parameters:
        - data_dir: Data Path
        - train_transform: Data augmentation operations for the training dataset.
        - val_transform: Data augmentation operations for the val dataset.
        - num_frames: The number of frames extracted from each video
        - mode: 'train' or 'val'。
        """
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.num_frames = num_frames
        self.video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
        self.labels = self._parse_labels()
        self.target_combinations = {(1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)}
        self.mode = mode

    def _parse_labels(self):
        labels = {}
        for video_file in self.video_files:
            parts = os.path.splitext(video_file)[0].split('_')
            labels[video_file] = (int(parts[1]), int(parts[2]))
        return labels

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.data_dir, video_file)
        frames = extract_and_process_frames(video_path, frame_count=self.num_frames)
        label = self.labels[video_file]

        pil_frames = [Image.fromarray(frame) for frame in frames]

        if self.mode == 'train':
            if (label in self.target_combinations):
                frames_tensor = torch.stack([self.train_transform(frame) for frame in pil_frames])
            else:
                frames_tensor = torch.stack([self.val_transform(frame) for frame in pil_frames])
        else:
            frames_tensor = torch.stack([self.val_transform(frame) for frame in pil_frames])

        return frames_tensor, label, video_file



class VideoDataset_test(Dataset):
    def __init__(self, data_dir, transform=None, num_frames=80):
        self.data_dir = data_dir
        self.transform = transform
        self.num_frames = num_frames
        self.video_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.mp4')],
            key=lambda x: int(x.split('_')[0]))
        
        self.labels = self._parse_labels()

    def __len__(self):
        return len(self.video_files)

    def _parse_labels(self):
        labels = {}
        for video_file in self.video_files:
            parts = os.path.splitext(video_file)[0].split('_')
            labels[video_file] = (int(parts[2]), int(parts[3]))
        return labels
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.data_dir, video_file)
        frames = self._extract_frames_sequential(video_path)
        label = self.labels[video_file]
        pil_frames = [Image.fromarray(frame) for frame in frames]
        frames_tensor = torch.stack([self.transform(frame) for frame in pil_frames])
        
        return frames_tensor.permute(0, 1, 2, 3), label, video_file

    def _extract_frames_sequential(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        interval = max(1, total_frames // self.num_frames)
        
        for i in range(self.num_frames):
            pos = min(i * interval, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames

    
def joint_loss(cls_loss, blinenum_pred, plinestate_pred, blinenum_label, plinestate_label, alpha=1.0, beta=1.0):
    # classification loss
    loss_bl = cls_loss(blinenum_pred, blinenum_label)
    loss_pl = cls_loss(plinestate_pred, plinestate_label)
    
    # Total Loss
    total_loss = alpha * loss_bl + beta * loss_pl
    
    return total_loss

def onehot_code(labels):
    blinenum_labels = labels[0]
    plinestate_labels = labels[1]
    
    blinenum_label_bs = len(blinenum_labels)
    blinenum_label_ns = 4
    blinenum_label = torch.zeros((blinenum_label_bs, blinenum_label_ns), dtype=torch.float32)
    blinenum_label.scatter_(1, blinenum_labels.unsqueeze(1).to(torch.int64), 1)

    plinestate_label_bs = len(plinestate_labels)
    plinestate_label_ns = 4
    plinestate_label = torch.zeros((plinestate_label_bs, plinestate_label_ns), dtype=torch.float32)
    plinestate_label.scatter_(1, plinestate_labels.unsqueeze(1).to(torch.int64), 1)

    score = labels[0].clone().detach().float() + labels[1].clone().detach().float()

    return blinenum_label, plinestate_label, score

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, EPOCHS, bl_wegiht, pl_wegiht, model_save_path, checkpoint_path=None):
    start_epoch = 0
    hist_loss = np.zeros(EPOCHS)
    hist_loss_val = np.zeros(EPOCHS)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        hist_loss[:start_epoch] = checkpoint['hist_loss']
        hist_loss_val[:start_epoch] = checkpoint['hist_loss_val']
        print(f"Loaded checkpoint from epoch {start_epoch}")

    val_loss_best = np.inf
    model.train()

    model_name = os.path.basename(model_save_path).replace('.pth', '')
    output_dir = folder_path
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f'{model_name}_prediction_results.txt')
    logging.basicConfig(filename=output_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

    for idx_epoch in range(start_epoch, EPOCHS):
        running_loss = 0
        with tqdm(total=len(train_dataloader.dataset), desc=f"[Epoch {idx_epoch + 1:3d}/{EPOCHS}]") as pbar:
            for batch, (video_clips, labels, video_name) in enumerate(train_dataloader):
                BS, N, C, H, W = video_clips.shape
                blinenum_pred, plinestate_pred, _, _ = model(video_clips.to(device))
                optimizer.zero_grad()
                
                blinenum_label, plinestate_label, score = onehot_code(labels=labels)

                loss = joint_loss(criterion, blinenum_pred, plinestate_pred, 
                                  blinenum_label.to(device), plinestate_label.to(device), 
                                  alpha=bl_wegiht, beta=pl_wegiht)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (batch + 1)})
                pbar.update(video_clips.shape[0])

            train_loss = running_loss / len(train_dataloader)
            val_loss = val(model, val_dataloader, criterion, device).item()
            pbar.set_postfix({'loss': train_loss, 'val_loss': val_loss})

            hist_loss[idx_epoch] = train_loss
            hist_loss_val[idx_epoch] = val_loss

            logging.info(f"Epoch {idx_epoch+1}/{EPOCHS} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                torch.save(model.state_dict(), model_save_path)

    logging.info(f"model exported to {model_save_path} with loss {val_loss_best:5f}")

    model.load_state_dict(torch.load(model_save_path))
    return model

def val(model, val_dataloader, criterion, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for batch, (video_clips, labels, video_name) in enumerate(val_dataloader):
            BS, N, C, H, W = video_clips.shape
            blinenum_pred, plinestate_pred, _ ,_ = model(video_clips.to(device))
            
            blinenum_label, plinestate_label, score  = onehot_code(labels=labels)

            loss = joint_loss(criterion, blinenum_pred, plinestate_pred, 
                    blinenum_label.to(device), plinestate_label.to(device), 
                    alpha=bl_wegiht, beta=pl_wegiht)

            running_loss += loss

    return running_loss / len(val_dataloader)

def predict(fold, first_pth, model_path, test_dataloader, device):
    model.load_state_dict(torch.load(model_path))

    model.eval()
    blinenum_accuracy = 0
    plinestate_accuracy = 0
    
    model_name = os.path.basename(first_pth).replace('.pth', '')
    output_dir = folder_path
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f'{model_name}_prediction_results.txt')

    blinenum_true_labels = [] # bline Ground Truth
    blinenum_pred_labels = [] # bline Predicted
    blinenum_pred_probs = []  # bline Predicting Probability
    plinestate_true_labels = [] # pline Ground Truth
    plinestate_pred_labels = [] # pline Predicted
    plinestate_pred_probs = []  # pline Predicting Probability

    b_all_fc = torch.tensor([]).to(device)
    p_all_fc = torch.tensor([]).to(device)
    b_all_target = torch.tensor([]).to(device)
    p_all_target = torch.tensor([]).to(device)

    with open(output_file_path, 'a') as f:
        f.write(f"Fold {fold + 1} : {model_path} predict result====>\n\n")
        with torch.no_grad():
            for batch, (video_clips, labels, video_name) in enumerate(test_dataloader):
                BS, N, C, H, W = video_clips.shape
                blinenum_pred, plinestate_pred, b_vid, p_vid = model(video_clips.to(device))
                b_all_fc = torch.cat((b_all_fc, b_vid), dim=0)
                p_all_fc = torch.cat((p_all_fc, p_vid), dim=0)

                blinenum_pred = torch.softmax(blinenum_pred, dim=1).to(device)
                plinestate_pred = torch.softmax(plinestate_pred, dim=1).to(device)
                blinenum_label, plinestate_label, score  = onehot_code(labels=labels)
                blinenum_label = blinenum_label.to(device)
                plinestate_label = plinestate_label.to(device)

                blinenum_pred_probs.append(blinenum_pred.cpu().numpy())
                plinestate_pred_probs.append(plinestate_pred.cpu().numpy())

                f.write(f"=↓=↓=↓=↓=↓=↓=↓=↓=↓={video_name}=↓=↓=↓=↓=↓=↓=↓=↓=↓=\n")
                f.write("blinenum_label: {}\n".format(", ".join(f"{x:.4f}" for x in blinenum_label.cpu().numpy().flatten())))
                f.write("blinenum_pred: {}\n".format(", ".join(f"{x:.4f}" for x in blinenum_pred.cpu().numpy().flatten())))
                f.write("plinestate_label: {}\n".format(", ".join(f"{x:.4f}" for x in plinestate_label.cpu().numpy().flatten())))
                f.write("plinestate_pred: {}\n".format(", ".join(f"{x:.4f}" for x in plinestate_pred.cpu().numpy().flatten())))

                max_indices = torch.argmax(blinenum_pred, dim=1)
                predicted_categories = blinenum_label_mapping[max_indices.item()]
                label_max_indices = torch.argmax(blinenum_label, dim=1)
                label_categories = blinenum_label_mapping[label_max_indices.item()]
                f.write("B-Line number: Label:{}, Prediction:{}\n".format(label_categories, predicted_categories))
                blinenum_pred_labels.extend(max_indices.cpu().numpy())
                blinenum_true_labels.extend(label_max_indices.cpu().numpy())

                if max_indices == label_max_indices:
                    blinenum_accuracy += 1

                max_indices2 = torch.argmax(plinestate_pred, dim=1)
                predicted_categories2 = plinestate_label_mapping[max_indices2.item()]
                label_max_indices2 = torch.argmax(plinestate_label, dim=1)
                label_categories2 = plinestate_label_mapping[label_max_indices2.item()]
                f.write("P-Line state: Label:{}, Prediction:{}\n".format(label_categories2, predicted_categories2))
                plinestate_pred_labels.extend(max_indices2.cpu().numpy())
                plinestate_true_labels.extend(label_max_indices2.cpu().numpy())

                if max_indices2 == label_max_indices2:
                    plinestate_accuracy += 1
                f.write(f"=↑=↑=↑=↑=↑=↑=↑=↑=↑={video_name}=↑=↑=↑=↑=↑=↑=↑=↑=↑=\n")
                f.write(f"\n")

                
                b_all_target = torch.cat((b_all_target, label_max_indices), dim=0)
                p_all_target = torch.cat((p_all_target, label_max_indices2), dim=0)

            all_blinenum_accuracy = blinenum_accuracy / len(test_dataloader)
            all_plinestate_accuracy = plinestate_accuracy / len(test_dataloader)
            f.write(f"B-Line Number num / all: {blinenum_accuracy} / {len(test_dataloader)}, Pleural-Line State num / all: {plinestate_accuracy} / {len(test_dataloader)}\n")
            f.write(f"B-Line Number ACC: {all_blinenum_accuracy:.4f}%, Pleural-Line State ACC: {all_plinestate_accuracy:.4f}%\n")

        # list to numpy
        blinenum_pred_probs = np.concatenate(blinenum_pred_probs, axis=0)
        plinestate_pred_probs = np.concatenate(plinestate_pred_probs, axis=0)

        # Calculate F1 score and ROC-AUC
        blinenum_f1 = f1_score(blinenum_true_labels, blinenum_pred_labels, average='weighted')
        plinestate_f1 = f1_score(plinestate_true_labels, plinestate_pred_labels, average='weighted')
        blinenum_auc = roc_auc_score(blinenum_true_labels, blinenum_pred_probs, multi_class='ovo')
        plinestate_auc = roc_auc_score(plinestate_true_labels, plinestate_pred_probs, multi_class='ovo')
        # Calculate the specificity.
        blinenum_specificities = calculate_specificity(blinenum_true_labels, blinenum_pred_labels, num_classes=4)
        plinestate_specificities = calculate_specificity(plinestate_true_labels, plinestate_pred_labels, num_classes=4)
        
        blinenum_sens = recall_score(blinenum_true_labels, blinenum_pred_labels, average='macro')
        plinestate_sens = recall_score(plinestate_true_labels, plinestate_pred_labels, average='macro')


        f.write(f"B-Line Number F1 score: {blinenum_f1:.4f}, ROC-AUC: {blinenum_auc:.4f}, Sensitivity: {blinenum_sens:.4f}\n")
        f.write(f"Pleural-Line State F1 score: {plinestate_f1:.4f}, ROC-AUC: {plinestate_auc:.4f}, Sensitivity: {plinestate_sens:.4f}\n")
        formatted_blinenum = [f"{x:.4f}" for x in blinenum_specificities]
        f.write(f"B-Line Number Sensitivity: {formatted_blinenum}\n")
        formatted_plinestate = [f"{x:.4f}" for x in plinestate_specificities]
        f.write(f"Pleural-Line State Sensitivity: {formatted_plinestate}\n\n\n")

    blinenum_metrics = calculate_metrics(blinenum_true_labels, blinenum_pred_labels, blinenum_pred_probs, blinenum_label_mapping, average_type='macro')
    plinestate_metrics = calculate_metrics(plinestate_true_labels, plinestate_pred_labels, plinestate_pred_probs, plinestate_label_mapping, average_type='macro')
    print_metrics(blinenum_metrics, plinestate_metrics, output_file_path)
    
    print(f"B-Line Number F1 score: {blinenum_f1:.4f}, ROC-AUC: {blinenum_auc:.4f}, Sensitivity: {blinenum_sens:.4f}")
    print(f"Pleural-Line State F1 score: {plinestate_f1:.4f}, ROC-AUC: {plinestate_auc:.4f}, Sensitivity: {plinestate_sens:.4f}")

    return all_blinenum_accuracy, all_plinestate_accuracy, blinenum_f1, plinestate_f1, blinenum_auc, plinestate_auc, blinenum_sens, plinestate_sens


def vote_predictions(predictions, avg_prob):
    """
    voting. if tie vote，the predicted probability mean determines Results

    Args:
        predictions: shape (num_models, num_samples)
        avg_prob: shape (num_samples, num_classes)

    Returns:
        final_predictions: shape (num_samples,)
    """
    num_models, num_samples = predictions.shape
    final_predictions = []

    for sample_idx in range(num_samples):
        votes = predictions[:, sample_idx]
        vote_counts = Counter(votes)
        final_class = np.argmax(avg_prob[sample_idx])  # Select maximum avg prob.

        # vote count
        max_vote = max(vote_counts.values())
        candidates = [cls for cls, count in vote_counts.items() if count == max_vote]

        # maximum count
        if len(candidates) == 1 and candidates[0] == final_class:
                final_predictions.append(candidates[0])
        elif len(candidates) == 1 and candidates[0] != final_class:
                final_predictions.append(final_class)
        else:
            final_predictions.append(final_class)

    return np.array(final_predictions)


def voting_predict(first_pth, fold_weights, test_dataloader, device):
    # load all models
    models = []
    for weight in fold_weights:
        model = WaveLUS(**vidnet_kwargs).to(device)
        model.load_state_dict(torch.load(weight))
        model.eval()
        models.append(model)

    blinenum_all_preds = []
    plinestate_all_preds = []

    all_blinenum_true = []
    all_blinenum_pred = []
    all_blinenum_probs = []
    all_plinestate_true = []
    all_plinestate_pred = []
    all_plinestate_probs = []

    model_name = os.path.basename(first_pth).replace('.pth', '')
    output_dir = folder_path
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f'{model_name}_prediction_results.txt')

    b_all_fc_0 = torch.tensor([]).to(device)
    p_all_fc_0 = torch.tensor([]).to(device)
    b_all_target_0 = torch.tensor([]).to(device)
    p_all_target_0 = torch.tensor([]).to(device)
    b_all_fc_1 = torch.tensor([]).to(device)
    p_all_fc_1 = torch.tensor([]).to(device)
    b_all_target_1 = torch.tensor([]).to(device)
    p_all_target_1 = torch.tensor([]).to(device)
    b_all_fc_2 = torch.tensor([]).to(device)
    p_all_fc_2 = torch.tensor([]).to(device)
    b_all_target_2 = torch.tensor([]).to(device)
    p_all_target_2 = torch.tensor([]).to(device)
    b_all_fc_3 = torch.tensor([]).to(device)
    p_all_fc_3 = torch.tensor([]).to(device)
    b_all_target_3 = torch.tensor([]).to(device)
    p_all_target_3 = torch.tensor([]).to(device)
    b_all_fc_4 = torch.tensor([]).to(device)
    p_all_fc_4 = torch.tensor([]).to(device)
    b_all_target_4 = torch.tensor([]).to(device)
    p_all_target_4 = torch.tensor([]).to(device)

    with open(output_file_path, 'a') as f:
        f.write(f"5 Fold final predict result on test datasets====>\n\n")
        with torch.no_grad():            
            for batch, (video_clips, labels, video_name) in enumerate(test_dataloader):
                BS, N, C, H, W = video_clips.shape
                Frame = [N] * BS
                blinenum_preds_prob = []
                plinestate_preds_prob = []
                blinenum_preds_cls = []
                plinestate_preds_cls = []

                blinenum_label, plinestate_label, score  = onehot_code(labels=labels)
                blinenum_label = blinenum_label.to(device)
                plinestate_label = plinestate_label.to(device)
                label_max_indices = torch.argmax(blinenum_label, dim=1)
                label_categories = blinenum_label_mapping[label_max_indices.item()]
                label_max_indices2 = torch.argmax(plinestate_label, dim=1)
                label_categories2 = plinestate_label_mapping[label_max_indices2.item()]


                # Use each model 
                for model in models:
                    blinenum_pred, plinestate_pred, _, _ = model(video_clips.to(device))
                    blinenum_preds_prob.append(torch.softmax(blinenum_pred, dim=1).cpu().numpy())
                    plinestate_preds_prob.append(torch.softmax(plinestate_pred, dim=1).cpu().numpy())
                    blinenum_pred_id = torch.argmax(blinenum_pred, dim=1).cpu().numpy()
                    plinestate_pred_id = torch.argmax(plinestate_pred, dim=1).cpu().numpy()
                    blinenum_preds_cls.append(blinenum_pred_id)
                    plinestate_preds_cls.append(plinestate_pred_id)
                

                blinenum_avg_prob = np.mean(blinenum_preds_prob, axis=0)  # shape: (BS, num_classes)
                plinestate_avg_prob = np.mean(plinestate_preds_prob, axis=0)  # shape: (BS, num_classes)
                # Voting
                blinenum_final_pred = vote_predictions(np.array(blinenum_preds_cls), blinenum_avg_prob)
                plinestate_final_pred = vote_predictions(np.array(plinestate_preds_cls), plinestate_avg_prob)
                

                all_blinenum_true.extend(label_max_indices.cpu().numpy())
                all_blinenum_pred.extend(blinenum_final_pred)
                all_blinenum_probs.extend(blinenum_avg_prob)
                all_plinestate_true.extend(label_max_indices2.cpu().numpy())
                all_plinestate_pred.extend(plinestate_final_pred)
                all_plinestate_probs.extend(plinestate_avg_prob)


                f.write(f"=↓=↓=↓=↓=↓=↓=↓=↓=↓={video_name[0]}=↓=↓=↓=↓=↓=↓=↓=↓=↓=\n")
                f.write("B-Line State Prediction (FOLDs):\n")
                for i, (prob, cls) in enumerate(zip(blinenum_preds_prob, blinenum_preds_cls)):
                    f.write(f"Model {i + 1}: {', '.join(f'{x:.4f}' for x in prob.flatten())}")
                    f.write(f"{', '.join(f'{cls}')}\n")
                f.write("B-Line Number Prediction (Mean): {}\n".format(", ".join(f"{x:.4f}" for x in blinenum_avg_prob.flatten())))
                f.write("B-Line Number Prediction: Label:{}, Predict:{}\n".format(label_categories, blinenum_label_mapping[blinenum_final_pred[0]]))
                
                f.write("Pleural-Line State Prediction (FOLDs):\n")
                for i, (prob, cls) in enumerate(zip(plinestate_preds_prob, plinestate_preds_cls)):
                    f.write(f"Model {i + 1}: {', '.join(f'{x:.4f}' for x in prob.flatten())}")
                    f.write(f"{', '.join(f'{cls}')}\n")
                f.write("Pleural-Line State Prediction (Mean): {}\n".format(", ".join(f"{x:.4f}" for x in plinestate_avg_prob.flatten())))
                f.write("Pleural-Line State Prediction: Label:{}, Predict:{}\n".format(label_categories2, plinestate_label_mapping[plinestate_final_pred[0]]))
                f.write(f"=↑=↑=↑=↑=↑=↑=↑=↑=↑={video_name[0]}=↑=↑=↑=↑=↑=↑=↑=↑=↑=\n")
            

    all_blinenum_metrics = calculate_metrics(all_blinenum_true, all_blinenum_pred, np.vstack(all_blinenum_probs), blinenum_label_mapping, average_type='macro')
    all_plinestate_metrics = calculate_metrics(all_plinestate_true, all_plinestate_pred, np.vstack(all_plinestate_probs), plinestate_label_mapping, average_type='macro')
    print_metrics(all_blinenum_metrics, all_plinestate_metrics, output_file_path)
    print("Over!")

if __name__ == '__main__':
    EPOCHS = 30
    LR = 1e-4
    BATCH_SIZE = 4
    num_frames = 64
    bl_wegiht = 0.5
    pl_wegiht = 0.5
    FOLD = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    
    blinenum_label_mapping = {
        0: "0 Lines",
        1: "<4 Lines",
        2: "4-6 Lines",
        3: ">6 Lines",
    }
    plinestate_label_mapping = {
        0: "Normal",
        1: "Coarse",
        2: "Irregular",
        3: "Fragmented",
    }

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    one_dataset = VideoDataset(data_dir='/data/4classes0309_all/0', train_transform=train_transform, val_transform=val_transform, num_frames=num_frames)
    two_dataset = VideoDataset(data_dir='/data/4classes0309_all/1', train_transform=train_transform, val_transform=val_transform, num_frames=num_frames)
    three_dataset = VideoDataset(data_dir='/data/4classes0309_all/2', train_transform=train_transform, val_transform=val_transform, num_frames=num_frames)
    four_dataset = VideoDataset(data_dir='/data/4classes0309_all/3', train_transform=train_transform, val_transform=val_transform, num_frames=num_frames)

    # fold0
    fold0_train_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold0/train', train_transform=train_transform, val_transform=val_transform, num_frames=num_frames)
    fold0_val_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold0/val', train_transform=val_transform, val_transform=val_transform, num_frames=num_frames)
    fold0_train_dataloader = DataLoader(fold0_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    fold0_val_dataloader = DataLoader(fold0_val_dataset, batch_size=1, shuffle=False, num_workers=4)
    # fold1
    fold1_train_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold1/train', train_transform=train_transform, val_transform=val_transform, num_frames=num_frames)
    fold1_val_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold1/val', train_transform=val_transform, val_transform=val_transform, num_frames=num_frames)
    fold1_train_dataloader = DataLoader(fold1_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    fold1_val_dataloader = DataLoader(fold1_val_dataset, batch_size=1, shuffle=False, num_workers=4)
    # fold2
    fold2_train_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold2/train', train_transform=train_transform, val_transform=val_transform, num_frames=num_frames)
    fold2_val_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold2/val', train_transform=val_transform, val_transform=val_transform, num_frames=num_frames)
    fold2_train_dataloader = DataLoader(fold2_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    fold2_val_dataloader = DataLoader(fold2_val_dataset, batch_size=1, shuffle=False, num_workers=4)
    # fold3
    fold3_train_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold3/train', train_transform=train_transform, val_transform=val_transform, num_frames=num_frames)
    fold3_val_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold3/val', train_transform=val_transform, val_transform=val_transform, num_frames=num_frames)
    fold3_train_dataloader = DataLoader(fold3_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    fold3_val_dataloader = DataLoader(fold3_val_dataset, batch_size=1, shuffle=False, num_workers=4)
    # fold4
    fold4_train_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold4/train', train_transform=train_transform, val_transform=val_transform, num_frames=num_frames)
    fold4_val_dataset = VideoDataset(data_dir='/data/4classes0309_5fold_01/fold4/val', train_transform=val_transform, val_transform=val_transform, num_frames=num_frames)
    fold4_train_dataloader = DataLoader(fold4_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    fold4_val_dataloader = DataLoader(fold4_val_dataset, batch_size=1, shuffle=False, num_workers=4)

    intertest_dataset = VideoDataset_test(data_dir='/data/open', transform=val_transform, num_frames=num_frames)
    intertest_dataloader = DataLoader(intertest_dataset, batch_size=1, shuffle=False, num_workers=4)
    all_train = [fold0_train_dataloader, fold1_train_dataloader, fold2_train_dataloader, fold3_train_dataloader, fold4_train_dataloader]
    all_val = [fold0_val_dataloader, fold1_val_dataloader, fold2_val_dataloader, fold3_val_dataloader, fold4_val_dataloader]


    # save the evaluation metrics for each fold
    blinenum_acc_scores = []
    plinestate_acc_scores = []
    blinenum_f1_scores = []
    plinestate_f1_scores = []
    blinenum_auc_scores = []
    plinestate_auc_scores = []
    blinenum_sens_scores = []
    plinestate_sens_scores = []

    # save the model weights for each fold
    fold_weights = []
    first_pth = []

    for fold in range(FOLD):
        print(f"Fold {fold + 1}")
        # creat DataLoader
        train_dataloader = all_train[fold]
        val_dataloader = all_val[fold]

        # Initialize the model
        vidnet_kwargs = {
            'num_heads': 16,
            'num_out1': 4,
            'num_out2': 4,
            'drop_rate': 0.5,
            'device': device,
        }
        model = WaveLUS(**vidnet_kwargs).to(device)

        # each class sample number
        class_counts = [len(one_dataset), len(two_dataset), len(three_dataset), len(four_dataset)]
        alpha = 1.0 / (torch.tensor(class_counts, dtype=torch.float32) + 1e-8)
        alpha = alpha / alpha.sum()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = FocalLoss(alpha=alpha, gamma=3).to(device)

        # mkdir
        folder_path = 'outputs/ours/WaveLUS/'
        os.makedirs(folder_path, exist_ok=True)

        model_save_path = f'{folder_path}/WaveLUS_{BATCH_SIZE}bs_{num_frames}_blw{bl_wegiht}_plw{pl_wegiht}_224_model_fold{fold + 1}_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.pth'
        checkpoint_path = f'{folder_path}/checkpoint_fold{fold + 1}.pth'  # Path to checkpoint for this fold
        traned_model = train(model, train_dataloader, val_dataloader, criterion, optimizer, device, EPOCHS, bl_wegiht, pl_wegiht, model_save_path, checkpoint_path)
        first_pth.append(model_save_path)
        # predict and metrics
        all_blinenum_acc, all_plinestate_acc, blinenum_f1, plinestate_f1, blinenum_auc, plinestate_auc, blinenum_sens, plinestate_sens = predict(fold, first_pth[0], first_pth[fold], intertest_dataloader, device)
        blinenum_acc_scores.append(all_blinenum_acc)
        plinestate_acc_scores.append(all_plinestate_acc)
        blinenum_f1_scores.append(blinenum_f1)
        plinestate_f1_scores.append(plinestate_f1)
        blinenum_auc_scores.append(blinenum_auc)
        plinestate_auc_scores.append(plinestate_auc)
        blinenum_sens_scores.append(blinenum_sens)
        plinestate_sens_scores.append(plinestate_sens)

    model_name = os.path.basename(first_pth[0]).replace('.pth', '')
    output_file_path = os.path.join(folder_path, f'{model_name}_prediction_results.txt')
    with open(output_file_path, 'a') as f:
        f.write(f"Fold {FOLD} Mean and Std====>\n\n")
        # 5-Folds Mean & Std
        blinenum_acc_mean = np.mean(blinenum_acc_scores)
        blinenum_acc_std = np.std(blinenum_acc_scores)
        plinestate_acc_mean = np.mean(plinestate_acc_scores)
        plinestate_acc_std = np.std(plinestate_acc_scores)
        blinenum_f1_mean = np.mean(blinenum_f1_scores)
        blinenum_f1_std = np.std(blinenum_f1_scores)
        plinestate_f1_mean = np.mean(plinestate_f1_scores)
        plinestate_f1_std = np.std(plinestate_f1_scores)
        blinenum_auc_mean = np.mean(blinenum_auc_scores)
        blinenum_auc_std = np.std(blinenum_auc_scores)
        plinestate_auc_mean = np.mean(plinestate_auc_scores)
        plinestate_auc_std = np.std(plinestate_auc_scores)
        blinenum_sens_mean = np.mean(blinenum_sens_scores)
        blinenum_sens_std = np.std(blinenum_sens_scores)
        plinestate_sens_mean = np.mean(plinestate_sens_scores)
        plinestate_sens_std = np.std(plinestate_sens_scores)

        f.write(f"Mean ACC of B-line number={blinenum_acc_mean:.4f}, Std={blinenum_acc_std:.4f}\n")
        f.write(f"Mean ACC of P-line state={plinestate_acc_mean:.4f}, Std={plinestate_acc_std:.4f}\n")
        f.write(f"Mean F1 of B-line number={blinenum_f1_mean:.4f}, Std={blinenum_f1_std:.4f}\n")
        f.write(f"Mean F1 of P-line state={plinestate_f1_mean:.4f}, Std={plinestate_f1_std:.4f}\n")
        f.write(f"Mean AUC of B-line number={blinenum_auc_mean:.4f}, Std={blinenum_auc_std:.4f}\n")
        f.write(f"Mean AUC of P-line state={plinestate_auc_mean:.4f}, Std={plinestate_auc_std:.4f}\n")
        f.write(f"Mean Sensitivity of B-line number={blinenum_sens_mean:.4f}, Std={blinenum_sens_std:.4f}\n")
        f.write(f"Mean Sensitivity of P-line state={plinestate_sens_mean:.4f}, Std={plinestate_sens_std:.4f}\n")

    voting_predict(first_pth[0], first_pth, intertest_dataloader, device)