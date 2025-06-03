
import numpy as np
from Action_Masking.sets import Zonotope
from typing import Union, List
import torch as th
from types import SimpleNamespace as SN
import time

class ActionMasking:
    def __init__(self, args):
        self.args = args
        self.u_space = np.array(self.args.u_space)
        self.u_low = self.u_space[0]
        self.u_high = self.u_space[1]
        self.u_eq = (self.u_high + self.u_low) / 2
        self.safe_space = Zonotope(G=np.array(self.args.G),c=self.u_eq.reshape(-1,1),cuda=self.args.use_cuda)

        self.u_space_th = th.from_numpy(np.array(self.args.u_space))
        self.center_th = th.from_numpy(self.safe_space.c)
        
        max_a = self.u_high
        min_a = self.u_low 
        self.As_H_action_space_th = th.from_numpy(np.vstack([np.eye(self.args.n_actions), -np.eye(self.args.n_actions)]))
        self.As_d_action_space_th = th.from_numpy(np.hstack([max_a, -min_a]))


        if self.args.use_cuda:
           self.u_space_th  = self.u_space_th.cuda()
           self.center_th = self.center_th.cuda()
           self.As_H_action_space_th = self.As_H_action_space_th.cuda()
           self.As_d_action_space_th = self.As_d_action_space_th.cuda()

        self.u_low_th = self.u_space_th[0]
        self.u_high_th = self.u_space_th[1]
        self.u_eq_th = (self.u_high_th + self.u_low_th) / 2
        self.center_th_sqz = self.center_th.squeeze()

    def ray_masking_np(self, action):
        np_action = self.from_th_to_np(action)
        center = np.squeeze(self.safe_space.c)
        #batched version
        # center_batched = np.tile(center, (action.shape[0], action.shape[1], 1))
        # transformed_action = self._transform_action_space_fn(np_action)
        # max_a = self.u_high
        # min_a = self.u_low 
        # H_base = np.vstack([np.eye(self.args.n_actions), -np.eye(self.args.n_actions)])  # shape (2*dim, dim)
        # As_H_action_space = np.broadcast_to(H_base, np_action.shape[:-1] + H_base.shape)
        # max_a_expanded = np.broadcast_to(max_a, np_action.shape[:-1] + (self.args.n_actions,))  # shape (2, 2, 3)
        # min_a_expanded = np.broadcast_to(min_a, np_action.shape[:-1] + (self.args.n_actions,))  # shape (2, 2, 3)
        # As_d_action_space = np.concatenate([max_a_expanded, -min_a_expanded], axis=-1) 
        # alpha_action, idx_action = self.find_line_intersection_polytope_batched(
        #     transformed_action,
        #     center_batched,
        #     As_H_action_space,
        #     As_d_action_space
        # )
        # normalized_action = (transformed_action - center_batched) / np.linalg.norm(transformed_action - center_batched, axis=-1, keepdims=True)
        # boundary_point = self.safe_space.boundary_point(np.reshape(normalized_action, (normalized_action.shape[0], 1))).squeeze()
        # alpha_safe = np.linalg.norm(boundary_point - center)
        # ratio = alpha_action / alpha_safe
        # eps = 0.9999
        # action_final = center_batched + 1/ratio * eps * (transformed_action - center_batched)
        # return action_final

        ## This step doesn't work yet, as it is unclear what the min and max points of the zonotope are.
        # Transform action to full action space
        # transformed_action = self.transform_action(action)
        # Transform action to outer approximation of safe action space
        # min_As = np.min(safe_space, axis=0)
        # max_As = np.max(safe_space, axis=0)
        # transformed_action = self.transform_action_custom(action, np.array([min_As, max_As]))
        # TODO: Transformation from [-1,1] to physical box
        transformed_action = self._transform_action_space_fn(np_action)
        # ---- Determine inside ratio (distance to edge of action space) ----
        # The inside ratio gives the scaling factor with which the shifted action has to be multiplied
        # to reach the edge of the action space in the direction of the normal vector of the critical halfspace.
        # Find relevant outside borders
        # TODO: check if this are the physical limits - should be correct
        max_a = self.u_high
        min_a = self.u_low 
        # Build halfspace/polytope representation of action space
        # Create a matrix with
        # [[1, 0, ..., 0], [-1, 0, ..., 0], [0, 1, ..., 0], [0, -1, ..., 0], ..., [0, 0, ..., 1], [0, 0, ..., -1]]
        # for each dimension of the action space
        As_H_action_space = np.vstack([np.eye(transformed_action.shape[0]), -np.eye(transformed_action.shape[0])])
        # Create a vector with the maximum and minimum values of the action space
        # [max_a_1, -min_a_1, max_a_2, -min_a_2, ..., max_a_n, -min_a_n]
        As_d_action_space = np.hstack([max_a, -min_a])
        # Find intersection points for action space
        alpha_action, idx_action = self.find_line_intersection_polytope(
            transformed_action,
            center,
            As_H_action_space,
            As_d_action_space
        )
        # Find intersection points for safe action space
        normalized_action = (transformed_action - center) / np.linalg.norm(transformed_action - center)
        boundary_point = self.safe_space.boundary_point(np.reshape(normalized_action, (normalized_action.shape[0], 1))).squeeze()
        alpha_safe = np.linalg.norm(boundary_point - center)
        # Calculate ratio that brings the intersection point of the action space to the edge of the safe action space
        ratio = alpha_action / alpha_safe
        # Safety ratio to avoid floating point errors
        eps = 0.9999
        # Calculate new point
        action_final = center + 1/ratio * eps * (transformed_action - center)
        # DEBUG
        if not self.safe_space.contains_point(action_final[:, np.newaxis]):
            print('Action not in safe space')
        return self.from_np_to_th(action_final)
    

    def ray_masking(self, action):
        transformed_action = self._transform_action_space_fn_th(action)
        alpha_action, idx_action = self.find_line_intersection_polytope_batched(
            transformed_action,
            self.center_th_sqz,
            self.As_H_action_space_th,
            self.As_d_action_space_th
        )
        alpha_action = alpha_action.reshape(action.shape[0],action.shape[1],-1)
        # Find intersection points for safe action space
        normalized_action = (transformed_action - self.center_th_sqz) /th.norm(transformed_action - self.center_th_sqz, dim=-1, keepdim=True)
        normalized_action = normalized_action.unsqueeze(-1)
        boundary_point = self.safe_space.boundary_point_th(normalized_action).squeeze(-1)
        alpha_safe = th.norm(boundary_point - self.center_th_sqz, dim=-1, keepdim=True)
        # Calculate ratio that brings the intersection point of the action space to the edge of the safe action space
        ratio = alpha_action / alpha_safe
        # Safety ratio to avoid floating point errors
        eps = 0.9999
        # Calculate new point
        action_final = self.center_th_sqz + 1/ratio * eps * (transformed_action - self.center_th_sqz)
        action_final = self._transform_action_space_fn_th(action_final).to(th.float32)#make sure output is [-1,1]
        # DEBUG
        # if not safe_space.contains_point(action_final[:, np.newaxis]):
        #     print('Action not in safe space')
        return action_final
    
    
    def find_line_intersection_polytope_batched(self, p, c, H, d):
        """
        Batched version of find_line_intersection_polytope.

        Args:
            p (B, N, D): Endpoints.
            c (B, N, D): Startpoints (centers of polytopes).
            H (M, D): Halfspace matrix.
            d (M,): Halfspace vector.

        Returns:
            alpha (B, N): Smallest positive intersection distances.
            idx (B, N): Indices of intersecting halfspaces.
        """


        Hc = th.matmul(c, H.T) # Hc = H @ c
        n = p - c
        norm = th.norm(n, dim=-1, keepdim=True) #np.linalg.norm(p - c)
        n_unit = n / norm #(p - c) * 1 / np.linalg.norm(p - c)

        Hn = th.matmul(n_unit, H.T) #Hn = H @ n
        Hn_safe = th.where(Hn == 0, th.tensor(th.finfo(Hn.dtype).eps, device=Hn.device), Hn)
        alpha = (d - Hc) / Hn_safe  # (..., M)  alpha = (d - Hc) / Hn
        alpha = th.where(alpha < 0, th.inf, alpha)  # mask out negative values
        idx_min = th.argmin(alpha, axis=-1)
        alpha_min = th.gather(alpha, dim=-1, index=idx_min.unsqueeze(-1)).squeeze(-1)

        return alpha_min, idx_min


    def find_line_intersection_polytope(self, p, c, H, d):
        r"""Find the first positive intersection of a line with a polygone in halfspace/polytope representation.

        x = c + \alpha * (p - c)/||p - c||
        H * x = d
        min_i \alpha_i s.t. H_i * x = d_i, \alpha_i >= 0

        Args:
            p: Point
            c: Center of polytope
            H: Halfspace matrix
            d: Halfspace vector
        Returns:
            \alpha (double): Line length to intersection point
            idx (int): Index of halfspace that is intersected first
        """
        idx = 0
        Hc = H @ c
        n = (p - c) * 1 / np.linalg.norm(p - c)
        Hn = H @ n
        alpha = (d - Hc) / Hn
        alpha[alpha < 0] = np.inf
        idx = np.argmin(alpha)
        return alpha[idx], idx
    
    
    def from_np_to_th(self, np_array):
        tensor = th.from_numpy(np_array)
        if self.args.use_cuda:
           tensor = tensor.cuda()
        return tensor
    
    def from_th_to_np(self, tensor):
        return tensor.detach().cpu().numpy()
    
    def _transform_action_space_fn(self, action):
            """Convert action from sampled action to [u_min, u_max]."""
            # e.g. [-1,1] -> [u_min, u_max]
            return np.clip(
                ((action + 1) / 2) * (self.u_high - self.u_low) + self.u_low,
                self.u_low,
                self.u_high,
            )
    
    def _transform_action_space_fn_th(self, action):
        return th.clamp(
                    ((action + 1) / 2) * (self.u_high_th - self.u_low_th) + self.u_low_th,
                    min=self.u_low_th,
                    max=self.u_high_th,
                )



if __name__ == "__main__":
    action = np.array([[[0.3040, -0.7897,  0.2007],
                        [0.0482,  0.0083, -0.5078]],
                       [[-0.9203, -0.4212, -0.3605],
                        [ 1.2268,  0.7288, -0.9301]]])
    
    action_th = th.from_numpy(action).cuda()
    print(action_th.shape)

    args =SN()
    args.u_space = [[-1, -1, -1], [1, 1, 1]]
    args.G = [[1, 1, 1, 0], [1, -1, 0, 1], [1, -1, 0, 0]]
    args.use_cuda = True
    args.n_actions = 3
    masking = ActionMasking(args)
    # for i in range(action.shape[0]):
    #     for j in range(action.shape[1]):
    #         final_action = masking.ray_masking_np(action_th[i][j])
            # print(final_action)
   
    final_action = masking.ray_masking(action_th)
    print(action_th)
    print(final_action)

