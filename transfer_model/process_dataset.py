
import torch
import numpy as np
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import smplx
import trimesh
import pyrender
import imageio

def init_smplx_model():
    """Initialize the SMPL-X model with predefined settings."""
    body_model = smplx.SMPLX('.datasets/body_models/models/smplx',
                             gender="neutral", 
                             create_body_pose=False, 
                             create_betas=False, 
                             create_global_orient=False, 
                             create_transl=False,
                             create_expression=True,
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
    return body_model

# Load SMPL-X parameters
param_path =  ".datasets/HuGe100K/test/samples/samples/images0/param/Algeria_female_buff_blazers_40~50 years old_279.npy"
param = np.load(param_path, allow_pickle=True).item()

# Extract SMPL-X parameters
smpl_params = param['smpl_params'].reshape(1, -1)
scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)

# Initialize SMPL-X model and generate vertices
device = torch.device("cpu")
model = init_smplx_model().to(device)
output = model(global_orient=global_orient, body_pose=pose, betas=betas, left_hand_pose=left_hand_pose,
               right_hand_pose=right_hand_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose,
               expression=expression)
vertices = output.vertices[0].detach().cpu().numpy()
faces = model.faces


# Create a Trimesh and Pyrender mesh
mesh = trimesh.Trimesh(vertices, faces)
mesh.export("tmp/smplx_output.obj")
print("Saved smplx_output.obj")

'''
mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
   
rendered_images_list = []

# Loop through multiple camera views
for idx in range(24):
    scene = pyrender.Scene()
    scene.add(mesh_pyrender)

    # Load and process camera parameters
    camera_params = param['poses']
    intrinsic_params = camera_params[idx][1]  # fx, fy, cx, cy
    extrinsic_params = camera_params[idx][0] # R|T

    # Set up Pyrender camera
    camera = pyrender.IntrinsicsCamera(fx=intrinsic_params[0], fy=intrinsic_params[1],
                                       cx=intrinsic_params[2], cy=intrinsic_params[3])

    # Convert COLMAP coordinates to Pyrender-compatible transformation
    extrinsic_params_inv = torch.inverse(extrinsic_params.clone())
    scale_factor = extrinsic_params_inv[:3, :3].norm(dim=1)
    extrinsic_params_inv[:3, 1:3] = -extrinsic_params_inv[:3, 1:3]
    extrinsic_params_inv[3, :3] = 0

    # Add camera and lighting
    scene.add(camera, pose=extrinsic_params_inv)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    scene.add(light, pose=extrinsic_params_inv)

    # Render the scene
    renderer = pyrender.OffscreenRenderer(640, 896)
    color, depth = renderer.render(scene)
    rendered_images_list.append(color)
    renderer.delete()

# Save rendered images as a video
rendered_images = np.stack(rendered_images_list)
imageio.mimwrite('rendered_results.mp4', rendered_images, fps=15)
print("Rendered results saved as rendered_results.mp4")

# Load an existing video and test alignment
video_path = param_path.replace("param", "videos").replace("npy", "mp4")
input_video = imageio.get_reader(video_path)
input_frames = [frame for frame in input_video]
blended_frames = [(0.5 * frame + 0.5 * render_frame).astype(np.uint8) for render_frame, frame in zip(rendered_images, input_frames)]
imageio.mimwrite('aligned_results.mp4', blended_frames, fps=15)
print("Blended video saved as aligned_results.mp4")

'''