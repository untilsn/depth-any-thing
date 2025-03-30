import argparse
import cv2
import os
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', action='store_true', help='Only display the prediction')
    parser.add_argument('--grayscale', action='store_true', help='Do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Read file list (Unicode support)
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('.txt'):
            with open(args.img_path, 'r', encoding='utf-8') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = [os.path.join(args.img_path, f) for f in os.listdir(args.img_path)]
    
    filenames = [os.path.abspath(f) for f in filenames]  # Convert to absolute paths

    os.makedirs(args.outdir, exist_ok=True)
    
    # Set default to grayscale instead of Spectral_r
    args.grayscale = True
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

        if raw_image is None:
            print(f"❌ Cannot open file: {filename}. Skipping this image.")
            continue

        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Properly handle unicode filenames
        base_name, ext = os.path.splitext(os.path.basename(filename))
        output_filename = os.path.join(args.outdir, f"{base_name}_Depth{ext}")
        
        # Use cv2.imwrite with proper encoding for Unicode filenames
        is_success, im_buf_arr = cv2.imencode(ext, depth)
        if is_success:
            im_buf_arr.tofile(output_filename)
        else:
            print(f"❌ Failed to save: {output_filename}")

    print("🎉 Processing completed!")