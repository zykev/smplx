# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

from typing import Optional, Tuple

import sys
import os
import os.path as osp

import numpy as np
from psbody.mesh import Mesh
import trimesh

import torch
from torch.utils.data import Dataset
from loguru import logger

import smplx
import pickle


class MeshFolder(Dataset):
    def __init__(
        self,
        data_folder: str,
        transforms=None,
        exts: Optional[Tuple] = None
    ) -> None:
        ''' Dataset similar to ImageFolder that reads meshes with the same
            topology
        '''
        if exts is None:
            exts = ['.npy']

        self.data_folder = osp.expandvars(data_folder)

        logger.info(
            f'Building mesh folder dataset for folder: {self.data_folder}')

        self.data_paths = []
        for item_name in sorted(os.listdir(os.path.join(self.data_folder, 'smplx'))):
            item_path = os.path.join(self.data_folder, 'smplx', item_name, 'smplx_param.pkl')
            self.data_paths.append(item_path)

        # self.data_paths = np.array([
        #     osp.join(self.data_folder, fname)
        #     for fname in os.listdir(self.data_folder)
        #     if any(fname.endswith(ext) for ext in exts)
        # ])
        self.num_items = len(self.data_paths)


        self.body_model = smplx.SMPLX('.datasets/body_models/models/smplx',
                             gender="neutral", 
                             create_body_pose=False, 
                             create_betas=False, 
                             create_global_orient=False, 
                             create_transl=False,
                             create_expression=False,
                             create_jaw_pose=True, 
                             create_leye_pose=True, 
                             create_reye_pose=True, 
                             create_right_hand_pose=False,
                             create_left_hand_pose=False,
                             use_pca=False,
                             num_pca_comps=12,
                             num_betas=10,
                             flat_hand_mean=False,
                             ext='pkl')

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):
        param_path = self.data_paths[index]

        smpl_params = pickle.load(open(param_path, 'rb'))

        # Extract SMPL-X parameters

        betas=smpl_params['betas']
        global_orient=smpl_params['global_orient']
        body_pose=smpl_params['body_pose']
        left_hand_pose=smpl_params['left_hand_pose']
        right_hand_pose=smpl_params['right_hand_pose']
        jaw_pose=smpl_params['jaw_pose']
        leye_pose=smpl_params['leye_pose']
        reye_pose=smpl_params['reye_pose']
        expression=smpl_params['expression']

        transl = smpl_params['transl']
        scale = smpl_params['scale']

        body = self.body_model(global_orient=global_orient, body_pose=body_pose, 
                                betas=betas, transl=transl,
                                left_hand_pose=left_hand_pose,
                                right_hand_pose=right_hand_pose, jaw_pose=jaw_pose, 
                                leye_pose=leye_pose, reye_pose=reye_pose,
                                expression=expression)
        
        vertices = (body.vertices[0] * scale).detach().cpu().numpy()
        faces = self.body_model.faces

        # Load the mesh
        # mesh = trimesh.load(mesh_path, process=False)

        return {
            'vertices': np.asarray(vertices, dtype=np.float32),
            'faces': np.asarray(faces, dtype=np.int32),
            'indices': index,
            'paths': param_path,
        }
