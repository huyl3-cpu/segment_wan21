import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
<<<<<<< HEAD
=======

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
import time
import requests
import uuid
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
from typing import List, Optional
<<<<<<< HEAD
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry
=======

# Import từ thư mục con
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry

# Import GroundingDINO dependencies
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob
<<<<<<< HEAD
logger = logging.getLogger('comfyui_segment_anything_a100')
_DEBUG_LICENSE = os.environ.get("LICENSE_DEBUG", "0") == "1"
_session_cache = {}
def validate_license(license_key: str, server_url: str, user_id: str = None, session_id: str = None, max_retries: int = 3) -> tuple[bool, str]:
    if not license_key or not license_key.strip():
        return False, "License key is empty"
    if not server_url or not server_url.strip():
        return False, "License server URL is empty"
    if not user_id:
        import platform
        import hashlib
        machine_id = platform.node() or "unknown"
        user_id = hashlib.md5(machine_id.encode()).hexdigest()[:16]
    if not session_id:
        session_id = user_id
        if _DEBUG_LICENSE:
            logger.info(f"[DEBUG] Session ID: {session_id}")
        else:
            logger.info("Session initialized (device-based)")
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
=======

logger = logging.getLogger('comfyui_segment_anything_a100')

# Global session cache to reuse sessions across multiple runs
# Key: (user_id, license_key) -> session_id
_session_cache = {}

# License validation with retry mechanism
def validate_license(license_key: str, server_url: str, user_id: str = None, session_id: str = None, max_retries: int = 3) -> tuple[bool, str]:
    """
    Validate license key by calling license server API with retry mechanism.
    Returns: (is_valid, error_message)
    """
    if not license_key or not license_key.strip():
        return False, "License key is empty"
    
    if not server_url or not server_url.strip():
        return False, "License server URL is empty"
    
    # Generate user_id and session_id if not provided
    if not user_id:
        import platform
        import hashlib
        # Use machine-specific identifier
        machine_id = platform.node() or "unknown"
        user_id = hashlib.md5(machine_id.encode()).hexdigest()[:16]
    
    # Reuse session_id from cache if available
    if not session_id:
        cache_key = (user_id, license_key.strip())
        if cache_key in _session_cache:
            session_id = _session_cache[cache_key]
            logger.info(f"Reusing cached session: {session_id[:10]}...")
        else:
            session_id = str(uuid.uuid4())
            _session_cache[cache_key] = session_id
            logger.info(f"Created new session: {session_id[:10]}...")
    
    
    # Retry loop
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            # Call license server API
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
            url = server_url.rstrip('/') + '/api/validate'
            payload = {
                "key": license_key.strip(),
                "user_id": user_id,
                "session_id": session_id
            }
<<<<<<< HEAD
            logger.info(f"License validation attempt {attempt}/{max_retries}...")
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    if _DEBUG_LICENSE:
                        logger.info(f"[DEBUG] License validated: {license_key[:12]}... (attempt {attempt})")
                    else:
                        logger.info("✅ License validated successfully")
=======
            
            logger.info(f"License validation attempt {attempt}/{max_retries}...")
            
            # Longer timeout for unstable networks
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    logger.info(f"✅ License validated successfully: {license_key[:12]}... (attempt {attempt})")
                    
                    # Send heartbeat immediately to activate session
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                    try:
                        heartbeat_url = server_url.rstrip('/') + '/api/heartbeat'
                        heartbeat_payload = {
                            "session_id": session_id,
<<<<<<< HEAD
                            "key": license_key.strip(),
                            "user_id": user_id
                        }
                        hb_response = requests.post(heartbeat_url, json=heartbeat_payload, timeout=5)
                        if hb_response.status_code == 200:
                            logger.info("✅ Session activated")
                        else:
                            if _DEBUG_LICENSE:
                                logger.warning(f"[DEBUG] Heartbeat failed: {hb_response.status_code}")
                            else:
                                logger.warning("⚠️ Heartbeat failed (non-critical)")
                    except Exception as hb_error:
                        logger.warning(f"Heartbeat error (non-critical): {hb_error}")
=======
                            "key": license_key.strip()
                        }
                        hb_response = requests.post(heartbeat_url, json=heartbeat_payload, timeout=5)
                        if hb_response.status_code == 200:
                            logger.info(f"Session activated via heartbeat")
                        else:
                            logger.warning(f"Heartbeat failed with status {hb_response.status_code}, but validation succeeded")
                    except Exception as hb_error:
                        logger.warning(f"Heartbeat error (non-critical): {hb_error}")
                    
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                    return True, ""
                else:
                    error = data.get("error", "Invalid license")
                    logger.warning(f"License validation failed: {error}")
                    return False, error
            else:
                last_error = f"License server returned status {response.status_code}"
<<<<<<< HEAD
                if _DEBUG_LICENSE:
                    logger.warning(f"[DEBUG] Attempt {attempt}/{max_retries} failed: {last_error}")
                else:
                    logger.warning(f"License validation attempt {attempt}/{max_retries} failed")
                if response.status_code in [403, 500, 502, 503, 504]:
                    if attempt < max_retries:
                        import time
                        wait_time = attempt * 2
=======
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {last_error}")
                
                # Special handling for 403 with cached session - might be expired
                if response.status_code == 403 and cache_key in _session_cache:
                    logger.warning(f"Session might be expired/invalid, clearing cache for retry")
                    # Clear cached session so next retry gets a fresh one
                    del _session_cache[cache_key]
                    if attempt < max_retries:
                        import time
                        logger.info(f"Creating new session and retrying in 2s...")
                        time.sleep(2)
                        # Create new session for next attempt
                        session_id = str(uuid.uuid4())
                        _session_cache[cache_key] = session_id
                        payload["session_id"] = session_id
                        continue
                
                # Retry on server errors (5xx) or 403
                if response.status_code in [403, 500, 502, 503, 504]:
                    if attempt < max_retries:
                        import time
                        wait_time = attempt * 2  # Exponential backoff: 2s, 4s, 6s
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"All {max_retries} attempts failed: {last_error}")
                        return False, last_error
                else:
<<<<<<< HEAD
                    logger.error(last_error)
                    return False, last_error
=======
                    # Don't retry on 4xx errors (except 403)
                    logger.error(last_error)
                    return False, last_error
                
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        except requests.exceptions.Timeout:
            last_error = "License server timeout (15s)"
            logger.warning(f"Attempt {attempt}/{max_retries} timed out")
            if attempt < max_retries:
                logger.info(f"Retrying...")
                continue
            else:
                logger.error(last_error)
                return False, last_error
<<<<<<< HEAD
=======
                
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        except requests.exceptions.ConnectionError:
            last_error = "Cannot connect to license server"
            logger.warning(f"Attempt {attempt}/{max_retries} connection error")
            if attempt < max_retries:
                import time
                time.sleep(2)
                logger.info(f"Retrying...")
                continue
            else:
                logger.error(last_error)
                return False, last_error
<<<<<<< HEAD
=======
                
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        except Exception as e:
            last_error = f"License validation error: {str(e)}"
            logger.error(f"Attempt {attempt}/{max_retries} exception: {last_error}")
            if attempt < max_retries:
                continue
            else:
                return False, last_error
<<<<<<< HEAD
    return False, last_error
=======
    
    # If we get here, all retries failed
    return False, last_error

# --- POLYFILL: Class xử lý Batch (Đã thêm thuộc tính .device) ---
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
class FastNestedTensor:
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask
<<<<<<< HEAD
=======

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return FastNestedTensor(cast_tensor, cast_mask)
<<<<<<< HEAD
    def decompose(self):
        return self.tensors, self.mask
    @property
    def device(self):
        return self.tensors.device
    @property
    def shape(self):
        return self.tensors.shape
def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
=======

    def decompose(self):
        return self.tensors, self.mask

    @property
    def device(self):
        return self.tensors.device

    @property
    def shape(self):
        return self.tensors.shape

def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    """
    Gom nhiều ảnh thành 1 Batch Tensor có Padding (thay thế hàm thư viện thiếu).
    """
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
    if tensor_list[0].ndim == 3:
        max_size = [0, 0, 0]
        for img in tensor_list:
            for i, s in enumerate(img.shape):
                max_size[i] = max(max_size[i], s)
<<<<<<< HEAD
=======
        
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
<<<<<<< HEAD
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
        return FastNestedTensor(tensor, mask)
    else:
        raise ValueError("Tensor input must be 3D (C, H, W)")
=======
        
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
            
        return FastNestedTensor(tensor, mask)
    else:
        raise ValueError("Tensor input must be 3D (C, H, W)")

# --- CONFIG ---
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
sam_model_dir_name = "sams"
sam_model_list = {
    "sam_vit_h (2.56GB)": {"model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"},
    "sam_vit_l (1.25GB)": {"model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"},
    "sam_vit_b (375MB)": {"model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"},
    "sam_hq_vit_h (2.57GB)": {"model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"},
    "sam_hq_vit_l (1.25GB)": {"model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"},
    "sam_hq_vit_b (379MB)": {"model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"},
    "mobile_sam(39MB)": {"model_url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"}
}
<<<<<<< HEAD
=======

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}
<<<<<<< HEAD
=======

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        return comfy_bert_model_base
    return 'bert-base-uncased'
<<<<<<< HEAD
def list_sam_model(): return list(sam_model_list.keys())
def list_groundingdino_model(): return list(groundingdino_model_list.keys())
=======

def list_sam_model(): return list(sam_model_list.keys())
def list_groundingdino_model(): return list(groundingdino_model_list.keys())

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)
    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination: return destination
    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder): os.makedirs(folder)
    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        download_url_to_file(url, destination)
    return destination
<<<<<<< HEAD
=======

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
def load_sam_model(model_name):
    sam_checkpoint_path = get_local_filepath(sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_file_name = os.path.basename(sam_checkpoint_path)
    model_type = model_file_name.split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=comfy.model_management.get_torch_device())
    sam.eval()
    sam.model_name = model_file_name
    return sam
<<<<<<< HEAD
=======

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
def load_groundingdino_model(model_name):
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(groundingdino_model_list[model_name]["config_url"], groundingdino_model_dir_name)
    )
    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(get_local_filepath(groundingdino_model_list[model_name]["model_url"], groundingdino_model_dir_name))
    dino.load_state_dict(local_groundingdino_clean_state_dict(checkpoint['model']), strict=False)
    dino.to(device=comfy.model_management.get_torch_device())
    dino.eval()
    return dino
<<<<<<< HEAD
def groundingdino_predict_batch(dino_model, image_pil_list, prompt, threshold, dtype=torch.float32):
=======

# --- OPTIMIZED BATCH FUNCTIONS ---

def groundingdino_predict_batch(dino_model, image_pil_list, prompt, threshold, dtype=torch.float32):
    # 1. Prepare Transforms
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
<<<<<<< HEAD
=======
    
    # 2. Transform all images
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
    tensors_list = []
    for img in image_pil_list:
        t_img, _ = transform(img, None)
        tensors_list.append(t_img)
<<<<<<< HEAD
    device = comfy.model_management.get_torch_device()
    samples = nested_tensor_from_tensor_list(tensors_list).to(device)
    samples = FastNestedTensor(samples.tensors.to(dtype=dtype), samples.mask)
    caption = prompt.lower().strip()
    if not caption.endswith("."): caption = caption + "."
    captions_batch = [caption] * len(image_pil_list)
    with torch.no_grad():
        outputs = dino_model(samples, captions=captions_batch)
    prediction_logits = outputs["pred_logits"].sigmoid()
    prediction_boxes = outputs["pred_boxes"]
    batch_boxes_result = []
    for i in range(len(image_pil_list)):
        logits = prediction_logits[i]
        boxes = prediction_boxes[i]
        mask = logits.max(dim=1)[0] > threshold
        boxes_filt = boxes[mask].cpu()
=======
    
    # 3. Create NestedTensor (Batch container for DINO)
    device = comfy.model_management.get_torch_device()
    samples = nested_tensor_from_tensor_list(tensors_list).to(device)
    # Convert to target precision
    samples = FastNestedTensor(samples.tensors.to(dtype=dtype), samples.mask)
    
    # 4. Prepare Captions
    caption = prompt.lower().strip()
    if not caption.endswith("."): caption = caption + "."
    captions_batch = [caption] * len(image_pil_list)
    
    # 5. Run Inference
    with torch.no_grad():
        outputs = dino_model(samples, captions=captions_batch)
    
    # 6. Process Outputs
    prediction_logits = outputs["pred_logits"].sigmoid()
    prediction_boxes = outputs["pred_boxes"]
    
    batch_boxes_result = []
    
    for i in range(len(image_pil_list)):
        logits = prediction_logits[i]
        boxes = prediction_boxes[i]
        
        mask = logits.max(dim=1)[0] > threshold
        boxes_filt = boxes[mask].cpu()
        
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        H, W = image_pil_list[i].size[1], image_pil_list[i].size[0]
        for j in range(boxes_filt.size(0)):
            boxes_filt[j] = boxes_filt[j] * torch.Tensor([W, H, W, H])
            boxes_filt[j][:2] -= boxes_filt[j][2:] / 2
            boxes_filt[j][2:] += boxes_filt[j][:2]
<<<<<<< HEAD
        batch_boxes_result.append(boxes_filt)
    return batch_boxes_result
=======
            
        batch_boxes_result.append(boxes_filt)
        
    return batch_boxes_result

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
def prepare_sam_batch(images_tensor, target_length=1024, dtype=torch.float32):
    B, H, W, C = images_tensor.shape
    device = comfy.model_management.get_torch_device()
    scale = target_length * 1.0 / max(H, W)
    new_h, new_w = int(H * scale + 0.5), int(W * scale + 0.5)
<<<<<<< HEAD
    img = images_tensor.permute(0, 3, 1, 2).to(device)
    img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    img = img * 255.0
    img = img.to(dtype=dtype)
    return img
=======
    
    img = images_tensor.permute(0, 3, 1, 2).to(device)
    img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    img = img * 255.0
    # Convert to target precision
    img = img.to(dtype=dtype)
    return img

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
class SAMModelLoader_A100:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model_name": (list_sam_model(), )}}
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )
<<<<<<< HEAD
    DISPLAY_NAME = "SamLoader"
    def main(self, model_name): return (load_sam_model(model_name), )
=======
    def main(self, model_name): return (load_sam_model(model_name), )

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
class GroundingDinoModelLoader_A100:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model_name": (list_groundingdino_model(), )}}
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )
<<<<<<< HEAD
    DISPLAY_NAME = "Segment Loader"
    def main(self, model_name): return (load_groundingdino_model(model_name), )
=======
    def main(self, model_name): return (load_groundingdino_model(model_name), )

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
class GroundingDinoSAMSegment_A100:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
<<<<<<< HEAD
                "sam_model_name": (list_sam_model(), {"tooltip": "SAM model to use for segmentation"}),
                "grounding_dino_model_name": (list_groundingdino_model(), {"tooltip": "GroundingDINO model to use for detection"}),
=======
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
<<<<<<< HEAD
=======
                "target_resolution": ("INT", {
                    "default": 1024, 
                    "min": 512, 
                    "max": 2048, 
                    "step": 64,
                    "tooltip": "SAM encoder resolution. Higher = better quality but slower. Default 1024."
                }),
                "precision": (["fp32", "fp16", "bf16"], {
                    "default": "fp32",
                    "tooltip": "Computation precision. fp16: ~2x faster, 50% VRAM | bf16: balanced (A100) | fp32: max accuracy"
                }),
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                "license_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "License key for validation. Leave empty to skip validation."
                }),
<<<<<<< HEAD
=======
                "license_server_url": ("STRING", {
                    "default": "https://license.xgroup-service.com",
                    "multiline": False,
                    "tooltip": "License server URL for validation."
                }),
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")
<<<<<<< HEAD
    DISPLAY_NAME = "SamSegment"
    def main(self, sam_model_name, grounding_dino_model_name, image, prompt, threshold, batch_size, target_resolution=1024, precision="fp32", license_key="", license_server_url="https://license.xgroup-service.com"):
        sam_model = load_sam_model(sam_model_name)
        grounding_dino_model = load_groundingdino_model(grounding_dino_model_name)
        total_images = image.shape[0]
        device = comfy.model_management.get_torch_device()
=======
    
    def main(self, grounding_dino_model, sam_model, image, prompt, threshold, batch_size, target_resolution=1024, precision="fp32", license_key="", license_server_url="https://license.xgroup-service.com"):
        total_images = image.shape[0]
        device = comfy.model_management.get_torch_device()
        
        # License validation
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        license_valid = True
        if license_key and license_key.strip():
            is_valid, error_msg = validate_license(license_key, license_server_url)
            if not is_valid:
                logger.error(f"⚠️ LICENSE INVALID: {error_msg}")
                license_valid = False
            else:
                logger.info("✅ License validated successfully")
        else:
            logger.warning("⚠️ No license key provided - running in demo mode")
            license_valid = False
<<<<<<< HEAD
=======
        
        # Apply precision mode to models
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }
        target_dtype = dtype_map.get(precision, torch.float32)
<<<<<<< HEAD
=======
        
        # Convert models to target precision
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        if precision != "fp32" and torch.cuda.is_available():
            sam_model = sam_model.to(dtype=target_dtype)
            grounding_dino_model = grounding_dino_model.to(dtype=target_dtype)
            logger.info(f"Models converted to {precision.upper()} precision")
<<<<<<< HEAD
        start_time = time.time()
        total_batches = (total_images + batch_size - 1) // batch_size
        logger.info(f"Starting segmentation: {total_images} images, batch_size={batch_size}, {total_batches} batches, resolution={target_resolution}, precision={precision.upper()}")
        sam_is_hq = False
        if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name: sam_is_hq = True
        predictor = SamPredictorHQ(sam_model, sam_is_hq)
        res_images, res_masks = [], []
        current_batch_size = batch_size
=======
        
        # Progress tracking setup
        start_time = time.time()
        total_batches = (total_images + batch_size - 1) // batch_size
        logger.info(f"Starting segmentation: {total_images} images, batch_size={batch_size}, {total_batches} batches, resolution={target_resolution}, precision={precision.upper()}")
        
        sam_is_hq = False
        if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name: sam_is_hq = True
        predictor = SamPredictorHQ(sam_model, sam_is_hq)
        
        res_images, res_masks = [], []
        
        # Current effective batch size (may be reduced on OOM)
        current_batch_size = batch_size
        
        # --- BATCH PROCESSING WITH ERROR HANDLING ---
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        i = 0
        batch_idx = 0
        while i < total_images:
            batch_idx += 1
            end_idx = min(i + current_batch_size, total_images)
            chunk_images = image[i:end_idx]
<<<<<<< HEAD
            logger.info(f"Processing batch {batch_idx}/{total_batches} [images {i}-{end_idx-1}] (batch_size={current_batch_size})")
            try:
                sam_input_batch = prepare_sam_batch(chunk_images, target_length=target_resolution, dtype=target_dtype)
                batch_features, batch_interm = predictor.get_image_features(sam_input_batch)
                original_size = (chunk_images.shape[1], chunk_images.shape[2])
                input_size = (sam_input_batch.shape[2], sam_input_batch.shape[3])
=======
            
            logger.info(f"Processing batch {batch_idx}/{total_batches} [images {i}-{end_idx-1}] (batch_size={current_batch_size})")
            
            try:
                # 1. ENCODER (Batch)
                sam_input_batch = prepare_sam_batch(chunk_images, target_length=target_resolution, dtype=target_dtype)
                batch_features, batch_interm = predictor.get_image_features(sam_input_batch)
                
                original_size = (chunk_images.shape[1], chunk_images.shape[2])
                input_size = (sam_input_batch.shape[2], sam_input_batch.shape[3])
                
                # 2. GROUNDING DINO (Batch)
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                chunk_pil_list = []
                for j in range(chunk_images.shape[0]):
                    chunk_pil_list.append(
                        Image.fromarray(np.clip(255. * chunk_images[j].cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGB')
                    )
                batch_boxes = groundingdino_predict_batch(grounding_dino_model, chunk_pil_list, prompt, threshold, dtype=target_dtype)
<<<<<<< HEAD
                for idx_in_batch in range(chunk_images.shape[0]):
                    curr_image_tensor = chunk_images[idx_in_batch]
                    boxes = batch_boxes[idx_in_batch]
=======
                
                # 3. DECODER (Feature Injection)
                for idx_in_batch in range(chunk_images.shape[0]):
                    curr_image_tensor = chunk_images[idx_in_batch]
                    boxes = batch_boxes[idx_in_batch]
                    
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                    if boxes.shape[0] == 0:
                        empty_mask = torch.zeros((1, original_size[0], original_size[1]), dtype=torch.float32, device="cpu")
                        res_masks.append(empty_mask)
                        res_images.append(curr_image_tensor.unsqueeze(0))
                        continue
<<<<<<< HEAD
=======

                    # Slice Features
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                    curr_feat = batch_features[idx_in_batch].unsqueeze(0)
                    curr_interm = None
                    if sam_is_hq and batch_interm is not None:
                         curr_interm = [f[idx_in_batch].unsqueeze(0) for f in batch_interm]
<<<<<<< HEAD
                    predictor.set_precomputed_features(curr_feat, curr_interm, original_size, input_size)
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, original_size)
                    transformed_boxes = transformed_boxes.to(device)
=======

                    # Inject Features
                    predictor.set_precomputed_features(curr_feat, curr_interm, original_size, input_size)
                    
                    # Transform Boxes
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, original_size)
                    transformed_boxes = transformed_boxes.to(device)
                    
                    # Predict
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                    masks, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False
                    )
<<<<<<< HEAD
                    final_mask = torch.any(masks, dim=0).float().cpu()
                    res_masks.append(final_mask)
                    mask_3d = final_mask[0].unsqueeze(-1).repeat(1, 1, 3)
                    masked_image = curr_image_tensor * mask_3d
                    res_images.append(masked_image.unsqueeze(0))
                del batch_features, batch_interm, sam_input_batch
                if (batch_idx % 5 == 0 or batch_idx == total_batches) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                i = end_idx
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    total_batches = (total_images + current_batch_size - 1) // current_batch_size
                    logger.warning(f"CUDA OOM detected! Reducing batch_size to {current_batch_size} and retrying batch {batch_idx}")
                    batch_idx -= 1
                else:
                    logger.error(f"CUDA OOM with batch_size=1. Cannot process image {i}. Skipping.")
=======
                    
                    final_mask = torch.any(masks, dim=0).float().cpu()
                    res_masks.append(final_mask)
                    
                    # Memory optimization: use multiplication instead of clone + masking
                    mask_3d = final_mask[0].unsqueeze(-1).repeat(1, 1, 3)
                    masked_image = curr_image_tensor * mask_3d
                    res_images.append(masked_image.unsqueeze(0))
                
                # Memory optimization: cleanup large tensors
                del batch_features, batch_interm, sam_input_batch
                
                # Clear CUDA cache periodically to prevent fragmentation
                if (batch_idx % 5 == 0 or batch_idx == total_batches) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Successfully processed this batch, move to next
                i = end_idx
                
            except torch.cuda.OutOfMemoryError as e:
                # CUDA OOM - try to recover
                torch.cuda.empty_cache()
                
                if current_batch_size > 1:
                    # Reduce batch size and retry
                    current_batch_size = max(1, current_batch_size // 2)
                    total_batches = (total_images + current_batch_size - 1) // current_batch_size
                    logger.warning(f"CUDA OOM detected! Reducing batch_size to {current_batch_size} and retrying batch {batch_idx}")
                    # Don't increment i - retry same batch with smaller size
                    batch_idx -= 1  # Reset batch counter since we're retrying
                else:
                    # Already at batch_size=1 and still OOM - cannot recover
                    logger.error(f"CUDA OOM with batch_size=1. Cannot process image {i}. Skipping.")
                    # Add empty results for this image
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                    empty_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")
                    res_masks.append(empty_mask)
                    res_images.append(image[i].unsqueeze(0))
                    i += 1
<<<<<<< HEAD
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx} (images {i}-{end_idx-1}): {str(e)}")
=======
                    
            except Exception as e:
                # General error handling
                logger.error(f"Error processing batch {batch_idx} (images {i}-{end_idx-1}): {str(e)}")
                # Add empty results for failed images
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
                for j in range(i, end_idx):
                    empty_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")
                    res_masks.append(empty_mask)
                    res_images.append(image[j].unsqueeze(0))
                i = end_idx
<<<<<<< HEAD
        elapsed = time.time() - start_time
        if elapsed > 0:
            logger.info(f"Segmentation completed in {elapsed:.2f}s ({total_images/elapsed:.2f} images/sec)")
=======

        # Final logging
        elapsed = time.time() - start_time
        if elapsed > 0:
            logger.info(f"Segmentation completed in {elapsed:.2f}s ({total_images/elapsed:.2f} images/sec)")
        
        # Final VRAM/GPU cleanup
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU VRAM cleared after processing")
<<<<<<< HEAD
        if len(res_images) == 0:
            return (torch.zeros_like(image), torch.zeros((total_images, image.shape[1], image.shape[2])))
        if not license_valid:
            logger.warning("⚠️ Returning randomized masks due to invalid/missing license")
            fake_masks = torch.rand((total_images, image.shape[1], image.shape[2])) > 0.5
            fake_masks = fake_masks.float()
=======
        
        if len(res_images) == 0:
            return (torch.zeros_like(image), torch.zeros((total_images, image.shape[1], image.shape[2])))
        
        # License check: return incorrect masks if license invalid
        if not license_valid:
            logger.warning("⚠️ Returning randomized masks due to invalid/missing license")
            # Return random noisy masks instead of correct ones
            fake_masks = torch.rand((total_images, image.shape[1], image.shape[2])) > 0.5
            fake_masks = fake_masks.float()
            # Apply fake masks to images
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
            fake_images = []
            for i in range(total_images):
                fake_mask_3d = fake_masks[i].unsqueeze(-1).repeat(1, 1, 3)
                fake_img = image[i] * fake_mask_3d
                fake_images.append(fake_img.unsqueeze(0))
            return (torch.cat(fake_images, dim=0), fake_masks)
<<<<<<< HEAD
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))
=======
            
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
class InvertMask:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"mask": ("MASK",)}}
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)
    def main(self, mask): return (1.0 - mask,)
<<<<<<< HEAD
=======

>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"mask": ("MASK",)}}
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["boolean_number"]
    FUNCTION = "main"
    CATEGORY = "segment_anything"
<<<<<<< HEAD
    def main(self, mask): return (torch.all(mask == 0).int().item(), )
=======
    def main(self, mask): return (torch.all(mask == 0).int().item(), )
>>>>>>> 30c069db518457e51546cc305b9b9d4ef5cdac17
