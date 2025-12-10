import os
# Early suppression of noisy third-party warnings during import
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"^CLIP\.tokenizer$")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"^kornia\.feature\.lightglue$")
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_dynamic_prompt, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--ckpt_tag', type=str, default='', help='Optional tag appended to checkpoint filename to avoid overwrite')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic','ensemble'], default='dynamic', 
                        help='Prompt type: dynamic modality-task or ensemble photo templates')
    # LARA attention switch
    parser.add_argument('--enable_lara', dest='enable_lara', action='store_true', help='Enable LARA attention for segmentation tokens')
    parser.add_argument('--disable_lara', dest='enable_lara', action='store_false', help='Disable LARA attention for segmentation tokens')
    parser.set_defaults(enable_lara=True)
    # VL distillation switches
    parser.add_argument('--enable_vl_distill', action='store_true', help='Enable vision-language distillation on AC (CLS vs text)')
    parser.add_argument('--vl_beta', type=float, default=0.3, help='Weight β for VL distillation loss')
    parser.add_argument('--enable_vl_dyn_coop', action='store_true', help='Use dynamic prompt features for VL teacher (override static cache)')
    # Visualization/logging removed: keep minimal training/test interface
    args = parser.parse_args()

    setup_seed(args.seed)

    # pilot alpha search removed

    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    # Prepare static text feature cache for VL distill if enabled
    if args.enable_vl_distill:
        from utils import precompute_text_feature_cache_default
        precompute_text_feature_cache_default(clip_model, device)

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list, enable_lara=args.enable_lara).to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True

    # self-distillation removed: do not freeze any teacher adapters

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))


    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    # few-shot image augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    with torch.amp.autocast(device_type='cuda'), torch.no_grad():
        task = 'AS' if CLASS_INDEX[args.obj] > 0 else 'AC'
        if args.prompt_mode == 'dynamic':
            text_features = encode_text_with_dynamic_prompt(clip_model, args.obj, task, device)
        else:
            text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)
        # VL teacher features (default from static cache; optionally override by dynamic prompt)
        if args.enable_vl_distill:
            from utils import load_cached_text_features
            text_features_vl = load_cached_text_features(args.obj, 'AC', device, model=clip_model)
            if args.enable_vl_dyn_coop:
                text_features_vl = text_features

    best_result = 0

    for epoch in range(args.epoch):
        print('epoch ', epoch, ':')

        loss_list = []
        for (image, gt, label) in train_loader:
            image = image.to(device)
            with torch.amp.autocast(device_type='cuda'):
                _, seg_tokens_raw, det_tokens_raw = model(image)
                # prepare patch-only tokens for downstream losses
                seg_patch_tokens = [p[:, 1:, :] for p in seg_tokens_raw]
                det_patch_tokens = [p[:, 1:, :] for p in det_tokens_raw]
                # ensure batch dimension present for tokens
                B_cur = image.shape[0]
                seg_patch_tokens = [p if p.dim()==3 else p.unsqueeze(0).expand(B_cur, -1, -1) for p in seg_patch_tokens]
                det_patch_tokens = [p if p.dim()==3 else p.unsqueeze(0).expand(B_cur, -1, -1) for p in det_patch_tokens]

                # self-distillation removed

                # VL distillation (AC only): β * sum_{l∈{12,18,24}} [1 - cos(p_cls_l, text_ci)]
                vl_loss = torch.tensor(0.0, device=device)
                if args.enable_vl_distill:
                    # per-sample text feature (normal/abnormal)
                    B = image.shape[0]
                    # gather text_features_vl[:, lbl] for each sample -> [B, C]
                    tf_batched = text_features_vl.unsqueeze(0).expand(B, -1, -1)  # [B, C, 2]
                    idx_sel = label.to(tf_batched.device).long().view(B, 1, 1).expand(B, tf_batched.shape[1], 1)
                    text_ci = tf_batched.gather(2, idx_sel).squeeze(2)  # [B, C]
                    # layers to apply
                    vl_layers = [l for l in [12, 18, 24] if l in args.features_list]
                    for vl in vl_layers:
                        s_idx = args.features_list.index(vl)
                        p_cls = det_tokens_raw[s_idx][:, 0, :]  # [B, C]
                        # cosine similarity per sample
                        sim = F.cosine_similarity(p_cls, text_ci, dim=1)
                        vl_loss = vl_loss + (1.0 - sim).mean()
                    vl_loss = args.vl_beta * vl_loss

                
                # det loss
                det_loss = 0
                image_label = label.to(device)
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                if CLASS_INDEX[args.obj] > 0:
                    # pixel level
                    seg_loss = 0
                    mask = gt[:, 0, :, :].to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    # self-distillation removed from loss
                    loss = seg_loss + det_loss + vl_loss
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    # AC only
                    loss = det_loss + vl_loss
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

                # visualization/logging removed

                loss_list.append(loss.item())

        print("Loss: ", np.mean(loss_list))


        seg_features = []
        det_features = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                # keep batch and remove cls token
                seg_patch_tokens = [p[:, 1:, :].contiguous() for p in seg_patch_tokens]
                det_patch_tokens = [p[:, 1:, :].contiguous() for p in det_patch_tokens]
                seg_features.append(seg_patch_tokens)
                det_features.append(det_patch_tokens)
        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
        

        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        if result > best_result:
            best_result = result
            print("Best result\n")
            if args.save_model == 1:
                # ensure directory exists
                os.makedirs(args.save_path, exist_ok=True)
                tag_suffix = ("_" + args.ckpt_tag) if getattr(args, 'ckpt_tag', '') else ''
                ckp_path = os.path.join(args.save_path, f'{args.obj}_s{args.shot}{tag_suffix}.pth')
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                            'det_adapters': model.det_adapters.state_dict()}, 
                            ckp_path)
          


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features, return_metrics=False):
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few= []

    # flatten memory features per layer to [M, C]
    seg_mem_flat = [m.reshape(-1, m.shape[-1]) if m.dim() == 3 else m for m in seg_mem_features]
    det_mem_flat = [m.reshape(-1, m.shape[-1]) if m.dim() == 3 else m for m in det_mem_features]

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        B = image.shape[0]

        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:

                # few-shot, seg head (per-sample)
                for b in range(B):
                    anomaly_maps_few_shot = []
                    for idx, p in enumerate(seg_patch_tokens):
                        cos = cos_sim(seg_mem_flat[idx], p[b])  # [M, L]
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(anomaly_map_few_shot,
                                                              size=args.img_size, mode='bilinear', align_corners=True)
                        # append 2D map [H, H] to avoid extra dim broadcasting
                        anomaly_maps_few_shot.append(anomaly_map_few_shot[0, 0].cpu().numpy())
                    score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                    seg_score_map_few.append(score_map_few)

                # zero-shot, seg head (per-sample)
                for b in range(B):
                    anomaly_maps = []
                    for layer in range(len(seg_patch_tokens)):
                        p = seg_patch_tokens[layer][b]  # [L, C]
                        p /= p.norm(dim=-1, keepdim=True)
                        anomaly = (100.0 * p @ text_features)  # [L, 2]
                        L = anomaly.shape[0]
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly.permute(1, 0).view(1, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[0, 1, :, :]  # [H, H]
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                    score_map_zero = np.sum(anomaly_maps, axis=0)
                    seg_score_map_zero.append(score_map_zero)
            
            else:
                # few-shot, det head (per-sample)
                for b in range(B):
                    anomaly_maps_few_shot = []
                    for idx, p in enumerate(det_patch_tokens):
                        cos = cos_sim(det_mem_flat[idx], p[b])
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(anomaly_map_few_shot,
                                                              size=args.img_size, mode='bilinear', align_corners=True)
                        # append 2D map [H, H] for consistency
                        anomaly_maps_few_shot.append(anomaly_map_few_shot[0, 0].cpu().numpy())
                    anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                    score_few_det = anomaly_map_few_shot.mean()
                    det_image_scores_few.append(score_few_det)

                # zero-shot, det head (batched)
                anomaly_score_b = torch.zeros(B, device=device)
                for layer in range(len(det_patch_tokens)):
                    p = det_patch_tokens[layer]
                    p /= p.norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * p @ text_features)  # [B, L, 2]
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]  # [B, L]
                    anomaly_score_b += anomaly_map.mean(dim=1)  # [B]
                det_image_scores_zero.extend(anomaly_score_b.cpu().numpy().tolist())

            # ground-truth accumulation per sample
            for b in range(B):
                gt_mask_list.append(mask[b].squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)


    if CLASS_INDEX[args.obj] > 0:

        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min() + 1e-8)
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min() + 1e-8)
    
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')

        if return_metrics:
            return {'seg_pAUC': float(seg_roc_auc), 'img_AUC': float(roc_auc_im)}
        return seg_roc_auc + roc_auc_im

    else:

        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)
    
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

        if return_metrics:
            return {'seg_pAUC': None, 'img_AUC': float(img_roc_auc_det)}
        return img_roc_auc_det


# Pilot alpha sweep for few-shot

# run_pilot_alpha_search_few removed


if __name__ == '__main__':
    main()


