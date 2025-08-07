# *************************************************************************
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************


import torch
from typing import List
import torch.nn.functional as F


def calculate_l1_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Calculate the L1 (Manhattan) distance between two tensors."""
    return torch.sum(torch.abs(tensor1 - tensor2), dim=1)


def calculate_l2_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Calculate the L2 (Euclidean) distance between two tensors."""
    return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim=1))


def calculate_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Calculate the Cosine Similarity between two tensors."""
    numerator = torch.sum(tensor1 * tensor2, dim=1)
    denominator = torch.sqrt(torch.sum(tensor1 ** 2, dim=1)) * torch.sqrt(torch.sum(tensor2 ** 2, dim=1))
    return numerator / denominator


def get_neighboring_patch(tensor: torch.Tensor, center: tuple, radius: int) -> torch.Tensor:
    """Get a neighboring patch from a tensor centered at a specific point."""
    r1, r2 = int(center[0]) - radius, int(center[0]) + radius + 1
    c1, c2 = int(center[1]) - radius, int(center[1]) + radius + 1
    
    return tensor[:, :, r1:r2, c1:c2]


def m_get_neighboring_patch(tensor: torch.Tensor, center: tuple, radius: int) -> torch.Tensor:
    """從特定中心點取得鄰近 patch，若越界則自動裁剪並 zero-padding 補足"""
    h, w = tensor.shape[2], tensor.shape[3]
    cy, cx = int(center[0]), int(center[1])
    
    # 期望的 patch 範圍
    r1, r2 = cy - radius, cy + radius + 1
    c1, c2 = cx - radius, cx + radius + 1

    # 計算實際可取的範圍（剪裁）
    r1_clip, r2_clip = max(0, r1), min(h, r2)
    c1_clip, c2_clip = max(0, c1), min(w, c2)

    patch = tensor[:, :, r1_clip:r2_clip, c1_clip:c2_clip]

    # 計算需要 padding 的上下左右
    pad_top = r1_clip - r1
    pad_bottom = r2 - r2_clip
    pad_left = c1_clip - c1
    pad_right = c2 - c2_clip

    # padding 格式為 (左, 右, 上, 下)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    patch = F.pad(patch, padding, mode='constant', value=0)

    return patch

def update_handle_points(handle_points: torch.Tensor, all_dist: torch.Tensor, r2) -> torch.Tensor:
    """Update handle points based on computed distances."""
    row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
    updated_point = torch.tensor([
        handle_points[0] - r2 + row,
        handle_points[1] - r2 + col
    ])
    return updated_point

def update_handle_points_v2(
    handle_points: torch.Tensor,
    all_dist: torch.Tensor,
    r2,
    target_point: torch.Tensor = None
) -> torch.Tensor:
    """Update handle points using L1 distance + (optionally) bias toward target."""
    if target_point is not None:
        # 計算每個 pixel 相對於 target 的座標懲罰
        h, w = all_dist.shape
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pos = torch.stack([yy, xx], dim=-1).to(all_dist.device).float()
        target_rel = target_point - (handle_points - r2)
        target_rel = target_rel.to(all_dist.device).float()
        dist_to_target = torch.norm(pos - target_rel.view(1, 1, 2), dim=-1)

        # 輕微加權：引導選擇方向但不會過度干擾
        all_dist = all_dist + 0.01 * dist_to_target

    row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
    updated_point = torch.tensor([
        handle_points[0] - r2 + row,
        handle_points[1] - r2 + col
    ])
    return updated_point

def extract_patch_mean_feature(F: torch.Tensor, center: torch.Tensor, r: int = 1):
    """
    Extract mean feature from a (2r+1)x(2r+1) patch around `center`.
    F: [1, C, H, W]
    center: [2] tensor [y, x]
    """
    y, x = int(center[0]), int(center[1])
    patch = F[:, :, y - r:y + r + 1, x - r:x + r + 1]  # shape: [1, C, 2r+1, 2r+1]
    return patch.mean(dim=[2, 3])  # shape: [1, C]

def point_tracking(F0: torch.Tensor, F1: torch.Tensor, handle_points: List[torch.Tensor],
                   handle_points_init: List[torch.Tensor], target_points, r2, distance_type: str = 'l1') -> List[torch.Tensor]:
    """Track points between F0 and F1 tensors."""
    with torch.no_grad():
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]
            #f0 = extract_patch_mean_feature(F0, pi0, 1)  # Extract mean feature from F0 patch
            f0_expanded = f0.unsqueeze(dim=-1).unsqueeze(dim=-1)

            #F1_neighbor = get_neighboring_patch(F1, pi, r2)
            F1_neighbor = m_get_neighboring_patch(F1, pi, r2)

            # Switch case for different distance functions
            if distance_type == 'l1':
                all_dist = calculate_l1_distance(f0_expanded, F1_neighbor)
            elif distance_type == 'l2':
                all_dist = calculate_l2_distance(f0_expanded, F1_neighbor)
            elif distance_type == 'cosine':
                all_dist = -calculate_cosine_similarity(f0_expanded, F1_neighbor)  # Negative for minimization

            all_dist = all_dist.squeeze(dim=0)
            handle_points[i] = update_handle_points(pi, all_dist, r2)
            #handle_points[i] = update_handle_points_v2(
            #    pi, all_dist, r2,
            #    target_point=target_points[i],  # 若有 target 可傳入
            #)
            #print('handle_points[i]',handle_points[i])

    return handle_points


def point_tracking_clipdrag_v1(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   target_points,
                   r_p=3):
    #print('point tracking a: consider distance to target.')
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = max(0,int(pi[0])-r_p), min(max_r,int(pi[0])+r_p+1)
            c1, c2 = max(0,int(pi[1])-r_p), min(max_c,int(pi[1])+r_p+1)
  
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            ################################################################
            # import pdb;pdb.set_trace()
            x,y  = torch.range(r1,r2-1),torch.range(c1,c2-1)
            xx,yy = torch.meshgrid(x,y)
            points = torch.stack([xx.flatten(),yy.flatten()],dim=1)
            point_dist = torch.sum((points-target_points[i])**2,dim=1).reshape(r2-r1,c2-c1)
            threshold = torch.sum((handle_points[i]-target_points[i])**2)
            all_dist[point_dist>=threshold] = float('inf')
            if threshold <=1:
                all_dist[r_p,r_p] = threshold 
            ################################################################
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            # handle_points[i][0] = pi[0] - args.r_p + row
            # handle_points[i][1] = pi[1] - args.r_p + col
            handle_points[i][0] = r1 + row
            handle_points[i][1] = c1 + col
            dist = torch.sum((handle_points[i]-target_points[i])**2)
            print(f'{i}-th point pairs: handle_point:{handle_points[i]},target_point:{target_points[i]},dist:{dist}')

        return handle_points

def point_tracking_betterdrag(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   target_points,
                   r_p=3):
    print('point tracking a: consider distance to target.')
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]
            f0_vec = f0.squeeze(0) 

            r1, r2 = max(0,int(pi[0])-r_p), min(max_r,int(pi[0])+r_p+1)
            c1, c2 = max(0,int(pi[1])-r_p), min(max_c,int(pi[1])+r_p+1)
  
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            ################################################################
            # import pdb;pdb.set_trace()
            x,y  = torch.range(r1,r2-1),torch.range(c1,c2-1)
            xx,yy = torch.meshgrid(x,y)
            points = torch.stack([xx.flatten(),yy.flatten()],dim=1)
            point_dist = torch.sum((points-target_points[i])**2,dim=1).reshape(r2-r1,c2-c1)
            threshold = torch.sum((handle_points[i]-target_points[i])**2)
            all_dist[point_dist>=threshold] = float('inf')
            if threshold <=1:
                all_dist[r_p,r_p] = threshold 
            ################################################################
            z_i = f0_vec.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(F1.device)
            tracking_score = (F1_neighbor * z_i).sum(dim=1).squeeze(0)
            λ = 0.7
            final_score = λ * all_dist - (1 - λ) * tracking_score
            row, col = divmod(final_score.argmin().item(), final_score.shape[-1])
            # handle_points[i][0] = pi[0] - args.r_p + row
            # handle_points[i][1] = pi[1] - args.r_p + col
            handle_points[i][0] = r1 + row
            handle_points[i][1] = c1 + col
            dist = torch.sum((handle_points[i]-target_points[i])**2)
            print(f'{i}-th point pairs: handle_point:{handle_points[i]},target_point:{target_points[i]},dist:{dist}')

        return handle_points

import math
def point_tracking_draglora(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   target_points,
                   r_p=3,
                   draglora_fast = False):
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        minD = []
        for i in range(len(handle_points)):
            #pi0, pi, ti = handle_points_init[i], handle_points[i], target_points[i]
            pi0 = torch.tensor(handle_points_init[i], dtype=torch.float32, device=F0.device)
            pi = torch.tensor(handle_points[i], dtype=torch.float32, device=F0.device)
            ti = torch.tensor(target_points[i], dtype=torch.float32, device=F0.device)
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            # Neighbourhood Region
            r1, r2 = max(0,int(pi[0])-r_p), min(max_r,int(pi[0])+r_p+1)
            c1, c2 = max(0,int(pi[1])-r_p), min(max_c,int(pi[1])+r_p+1)
            x_coords = torch.arange(r1, r2, device=F0.device)
            y_coords = torch.arange(c1, c2, device=F0.device)
            x_grid, y_grid = torch.meshgrid(x_coords, y_coords)
            coordinates = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=1)
            coordinates = coordinates.to(ti.device)  # 確保在相同 device

            if draglora_fast:
                # Angle-closer Region
                diretcion = ti-pi 
                distance = diretcion.norm() 
                if distance<1: 
                    continue 
                diretcion  = diretcion / distance 
                cos_angle_threshold = math.cos(math.radians(30)) 
                vectors = coordinates - pi 
                vectors = vectors / (vectors.norm(dim=1,keepdim=True)+1e-8) 
                cos_angles = torch.sum(vectors * diretcion, dim=1) 
                validate_mask = cos_angles >= cos_angle_threshold 
                coordinates = coordinates[validate_mask] 
            else:
                # Distance-closer Region
                threshold = torch.norm(pi - ti) 
                point_distances = torch.norm(coordinates-ti,dim=1) 
                coordinates = coordinates[point_distances<=threshold] 

            F1_neighbor = F1[:, :, coordinates[:,0], coordinates[:,1]]

            all_dist = (f0.unsqueeze(dim=-1) - F1_neighbor).abs().mean(dim=1) ### pt_
            all_dist = all_dist.squeeze(dim=0)
            minD.append(round(all_dist.min().item(),2))

            new_point = coordinates[all_dist.argmin().item()]
            #handle_points[i][0] = new_point[0]
            #handle_points[i][1] = new_point[1]
            handle_points[i][0] = new_point[0].item()
            handle_points[i][1] = new_point[1].item()

        for i in range(len(handle_points)):
            handle_points[i] = torch.tensor(handle_points[i], dtype=torch.float32, device=F0.device)
            target_points[i] = torch.tensor(target_points[i], dtype=torch.float32, device=F0.device)

        return handle_points,tuple(minD)


def interpolate_feature_patch(feat: torch.Tensor, y: float, x: float, r: int) -> torch.Tensor:
    """Obtain the bilinear interpolated feature patch."""
    x0, y0 = torch.floor(x).long(), torch.floor(y).long()
    x1, y1 = x0 + 1, y0 + 1

    weights = torch.tensor([(x1 - x) * (y1 - y), (x1 - x) * (y - y0), (x - x0) * (y1 - y), (x - x0) * (y - y0)])
    weights = weights.to(feat.device)

    patches = torch.stack([
        feat[:, :, y0 - r:y0 + r + 1, x0 - r:x0 + r + 1],
        feat[:, :, y1 - r:y1 + r + 1, x0 - r:x0 + r + 1],
        feat[:, :, y0 - r:y0 + r + 1, x1 - r:x1 + r + 1],
        feat[:, :, y1 - r:y1 + r + 1, x1 - r:x1 + r + 1]
    ])

    return torch.sum(weights.view(-1, 1, 1, 1, 1) * patches, dim=0)

def interpolate_feature_patch_safe(feat: torch.Tensor, y: float, x: float, r: int) -> torch.Tensor:
    """
    安全版：取得插值 feature patch，具備邊界保護（zero padding）。
    """
    device = feat.device
    B, C, H, W = feat.shape

    # 取得周圍整數座標
    x0, y0 = int(torch.floor(x).item()), int(torch.floor(y).item())
    x1, y1 = x0 + 1, y0 + 1

    # 計算 bilinear 權重
    dx, dy = x - x0, y - y0
    weights = torch.tensor([
        (1 - dx) * (1 - dy),
        (1 - dx) * dy,
        dx * (1 - dy),
        dx * dy
    ], device=device)

    # 定義四個 patch 的中心點座標
    centers = [(y0, x0), (y1, x0), (y0, x1), (y1, x1)]

    patches = []
    for cy, cx in centers:
        # 裁剪範圍
        r1, r2 = cy - r, cy + r + 1
        c1, c2 = cx - r, cx + r + 1

        # 限制在合法範圍內
        r1_clip, r2_clip = max(0, r1), min(H, r2)
        c1_clip, c2_clip = max(0, c1), min(W, c2)

        patch = feat[:, :, r1_clip:r2_clip, c1_clip:c2_clip]

        # 補 0 padding
        pad_top = r1_clip - r1
        pad_bottom = r2 - r2_clip
        pad_left = c1_clip - c1
        pad_right = c2 - c2_clip

        patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        patches.append(patch)

    patches = torch.stack(patches, dim=0)  # shape: [4, B, C, r*2+1, r*2+1]
    result = torch.sum(weights.view(-1, 1, 1, 1, 1) * patches, dim=0)  # weighted sum

    return result  # shape: [B, C, r*2+1, r*2+1]


def check_handle_reach_target(handle_points: list, target_points: list) -> bool:
    """Check if handle points are close to target points."""
    all_dists = torch.tensor([(p - q).norm().item() for p, q in zip(handle_points, target_points)])
    return (all_dists < 2.0).all().item()
