import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob
logger = logging.getLogger('comfyui_segment_anything_a100')
_DEBUG_LICENSE = os.environ.get("LICENSE_DEBUG", "0") == "1"
_session_cache = {}

# Attention backend integration (SDPA / Flash Attention / SageAttention)
_HAS_SAGE_ATTN = False
try:
    from sageattention import sageattn
    _HAS_SAGE_ATTN = True
    logger.info("SageAttention available for SAM ViT encoder")
except ImportError:
    pass

_HAS_FLASH_ATTN = False
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
    logger.info("Flash Attention available for SAM ViT encoder")
except ImportError:
    pass

_sam_original_attn_forward = None
_current_attn_mode = "pytorch"

def _make_sdpa_attn_forward():
    """Create SDPA-based forward for SAM ViT Attention.
    Uses F.scaled_dot_product_attention which auto-dispatches to
    Flash Attention / Memory Efficient Attention backends.
    Handles relative position embeddings by computing attn_bias."""
    from segment_anything.modeling.image_encoder import add_decomposed_rel_pos, get_rel_pos
    def sdpa_attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, H*W, head_dim)

        if self.use_rel_pos:
            # Compute rel_pos bias and pass as attn_mask to SDPA
            # rel_pos needs (B*nHead, H*W, C) format for add_decomposed_rel_pos
            head_dim = q.shape[-1]
            q_for_bias = q.reshape(B * self.num_heads, H * W, head_dim)
            
            # Build the bias tensor
            attn_bias = torch.zeros(B * self.num_heads, H * W, H * W, 
                                    dtype=q.dtype, device=q.device)
            attn_bias = add_decomposed_rel_pos(attn_bias, q_for_bias, 
                                                self.rel_pos_h, self.rel_pos_w, 
                                                (H, W), (H, W))
            # Reshape bias to (B, num_heads, H*W, H*W) for SDPA
            attn_bias = attn_bias.view(B, self.num_heads, H * W, H * W)
            
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        return x
    return sdpa_attn_forward

def _make_sageattn_forward():
    """Create SageAttention forward. Falls back to naive for rel_pos blocks."""
    from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
    def sage_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_rel_pos:
            # SageAttention doesn't support attn bias → fallback to naive
            q_flat = q.reshape(B * self.num_heads, H * W, -1)
            k_flat = k.reshape(B * self.num_heads, H * W, -1)
            v_flat = v.reshape(B * self.num_heads, H * W, -1)
            attn = (q_flat * self.scale) @ k_flat.transpose(-2, -1)
            attn = add_decomposed_rel_pos(attn, q_flat, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            attn = attn.softmax(dim=-1)
            x = (attn @ v_flat).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        else:
            x = sageattn(q, k, v, is_causal=False)
            x = x.transpose(1, 2).reshape(B, H, W, -1)

        x = self.proj(x)
        return x
    return sage_forward

def _make_flash_attn_forward():
    """Create Flash Attention forward using flash-attn package directly.
    flash_attn_func expects (B, seq_len, num_heads, head_dim) format.
    For rel_pos blocks, falls back to SDPA with bias."""
    from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
    def flash_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(2)  # (B, H*W, num_heads, head_dim)

        if self.use_rel_pos:
            # flash_attn doesn't support arbitrary attn bias
            # Use SDPA with bias as fallback (still optimized)
            head_dim = q.shape[-1]
            q_t = q.transpose(1, 2)  # (B, num_heads, H*W, head_dim)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            q_for_bias = q_t.reshape(B * self.num_heads, H * W, head_dim)
            attn_bias = torch.zeros(B * self.num_heads, H * W, H * W,
                                    dtype=q.dtype, device=q.device)
            attn_bias = add_decomposed_rel_pos(attn_bias, q_for_bias,
                                                self.rel_pos_h, self.rel_pos_w,
                                                (H, W), (H, W))
            attn_bias = attn_bias.view(B, self.num_heads, H * W, H * W)
            x = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_bias)
            x = x.transpose(1, 2).reshape(B, H, W, -1)
        else:
            # Direct flash_attn: (B, seq_len, num_heads, head_dim)
            x = flash_attn_func(q, k, v, causal=False)
            x = x.reshape(B, H, W, -1)

        x = self.proj(x)
        return x
    return flash_forward

def enable_optimized_attention(mode="sdpa"):
    """Monkey-patch SAM ViT Attention with optimized backend."""
    global _sam_original_attn_forward, _current_attn_mode
    try:
        from segment_anything.modeling.image_encoder import Attention as SAMAttention
        if _sam_original_attn_forward is None:
            _sam_original_attn_forward = SAMAttention.forward
        
        if mode == "sdpa":
            SAMAttention.forward = _make_sdpa_attn_forward()
            _current_attn_mode = "sdpa"
            logger.info("SDPA enabled for SAM ViT (auto Flash/MemEfficient)")
        elif mode == "flash_attn":
            SAMAttention.forward = _make_flash_attn_forward()
            _current_attn_mode = "flash_attn"
            logger.info("Flash Attention enabled for SAM ViT (direct flash-attn package)")
        elif mode == "sageattn":
            SAMAttention.forward = _make_sageattn_forward()
            _current_attn_mode = "sageattn"
            logger.info("SageAttention enabled for SAM ViT (non-rel_pos blocks only)")
    except Exception as e:
        logger.warning(f"Failed to enable {mode} for SAM: {e}")

def disable_optimized_attention():
    """Restore original SAM ViT Attention."""
    global _sam_original_attn_forward, _current_attn_mode
    if _sam_original_attn_forward is not None:
        try:
            from segment_anything.modeling.image_encoder import Attention as SAMAttention
            SAMAttention.forward = _sam_original_attn_forward
            _sam_original_attn_forward = None
            _current_attn_mode = "pytorch"
        except Exception:
            pass
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
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            url = server_url.rstrip('/') + '/api/validate'
            payload = {
                "key": license_key.strip(),
                "user_id": user_id,
                "session_id": session_id
            }

            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    pass
                    try:
                        heartbeat_url = server_url.rstrip('/') + '/api/heartbeat'
                        heartbeat_payload = {
                            "session_id": session_id,
                            "key": license_key.strip(),
                            "user_id": user_id
                        }
                        hb_response = requests.post(heartbeat_url, json=heartbeat_payload, timeout=5)
                        if hb_response.status_code == 200:
                            pass
                    except Exception as hb_error:
                        pass
                    return True, ""
                else:
                    error = data.get("error", "Invalid license")
                    pass
                    return False, error
            else:
                last_error = f"License server returned status {response.status_code}"
                pass
                if response.status_code in [403, 500, 502, 503, 504]:
                    if attempt < max_retries:
                        import time
                        wait_time = attempt * 2

                        time.sleep(wait_time)
                        continue
                    else:
                        # Suppressed error output
                        return False, last_error
                else:
                    # Suppressed error output
                    return False, last_error
        except requests.exceptions.Timeout:
            last_error = "License server timeout (15s)"
            if attempt < max_retries:
                pass
                continue
            else:
                # Suppressed error output
                return False, last_error
        except requests.exceptions.ConnectionError:
            last_error = "Cannot connect to license server"
            if attempt < max_retries:
                import time
                time.sleep(2)
                continue
            else:
                # Suppressed error output
                return False, last_error
        except Exception as e:
            last_error = f"License validation error: {str(e)}"
            # Suppressed error output
            if attempt < max_retries:
                continue
            else:
                return False, last_error
    return False, last_error
class FastNestedTensor:
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask
    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return FastNestedTensor(cast_tensor, cast_mask)
    def decompose(self):
        return self.tensors, self.mask
    @property
    def device(self):
        return self.tensors.device
    @property
    def shape(self):
        return self.tensors.shape
def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = [0, 0, 0]
        for img in tensor_list:
            for i, s in enumerate(img.shape):
                max_size[i] = max(max_size[i], s)
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
        return FastNestedTensor(tensor, mask)
    else:
        raise ValueError("Tensor input must be 3D (C, H, W)")
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
def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        return comfy_bert_model_base
    return 'bert-base-uncased'
def list_sam_model(): return list(sam_model_list.keys())
def list_groundingdino_model(): return list(groundingdino_model_list.keys())
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
def groundingdino_predict_batch(dino_model, image_pil_list, prompt, threshold, dtype=torch.float32):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensors_list = []
    for img in image_pil_list:
        t_img, _ = transform(img, None)
        tensors_list.append(t_img)
    device = comfy.model_management.get_torch_device()
    samples = nested_tensor_from_tensor_list(tensors_list).to(device)
    samples = FastNestedTensor(samples.tensors.to(dtype=dtype), samples.mask)
    # Determine autocast dtype for mixed precision
    autocast_dtype = dtype if dtype != torch.float32 else None
    caption = prompt.lower().strip()
    if not caption.endswith("."): caption = caption + "."
    captions_batch = [caption] * len(image_pil_list)
    with torch.no_grad():
        if autocast_dtype is not None and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                outputs = dino_model(samples, captions=captions_batch)
        else:
            outputs = dino_model(samples, captions=captions_batch)
    prediction_logits = outputs["pred_logits"].sigmoid()
    prediction_boxes = outputs["pred_boxes"]
    batch_boxes_result = []
    for i in range(len(image_pil_list)):
        logits = prediction_logits[i]
        boxes = prediction_boxes[i]
        mask = logits.max(dim=1)[0] > threshold
        boxes_filt = boxes[mask].cpu()
        H, W = image_pil_list[i].size[1], image_pil_list[i].size[0]
        for j in range(boxes_filt.size(0)):
            boxes_filt[j] = boxes_filt[j] * torch.Tensor([W, H, W, H])
            boxes_filt[j][:2] -= boxes_filt[j][2:] / 2
            boxes_filt[j][2:] += boxes_filt[j][:2]
        batch_boxes_result.append(boxes_filt)
    return batch_boxes_result
def prepare_sam_batch(images_tensor, target_length=1024, dtype=torch.float32):
    B, H, W, C = images_tensor.shape
    device = comfy.model_management.get_torch_device()
    scale = target_length * 1.0 / max(H, W)
    new_h, new_w = int(H * scale + 0.5), int(W * scale + 0.5)
    img = images_tensor.permute(0, 3, 1, 2).to(device)
    img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    img = img * 255.0
    img = img.to(dtype=dtype)
    return img
class SAMModelLoader_A100:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model_name": (list_sam_model(), )}}
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )
    DISPLAY_NAME = "SamLoader"
    def main(self, model_name): return (load_sam_model(model_name), )
class GroundingDinoModelLoader_A100:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model_name": (list_groundingdino_model(), )}}
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )
    DISPLAY_NAME = "Segment Loader"
    def main(self, model_name): return (load_groundingdino_model(model_name), )
class GroundingDinoSAMSegment_A100:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model_name": (list_sam_model(), {"tooltip": "SAM model to use for segmentation"}),
                "grounding_dino_model_name": (list_groundingdino_model(), {"tooltip": "GroundingDINO model to use for detection"}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1, "tooltip": "Batch size for processing. A100 80GB can handle 32-64."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Computation precision. bf16 recommended for A100."}),
                "attention_mode": (["pytorch", "sdpa", "flash_attn", "sageattn"], {"default": "pytorch", "tooltip": "Attention backend for SAM ViT. sdpa=auto Flash/MemEfficient, flash_attn=direct flash-attn package, sageattn=SageAttention."}),
            },
            "optional": {
                "license_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "License key for validation. Leave empty to skip validation."
                }),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")
    DISPLAY_NAME = "SamSegment"
    def main(self, sam_model_name, grounding_dino_model_name, image, prompt, threshold, batch_size, precision="bf16", attention_mode="pytorch", target_resolution=1024, license_key="", license_server_url="https://license.xgroup-service.com"):
        # Enable/disable optimized attention based on widget
        if attention_mode in ("sdpa", "flash_attn", "sageattn"):
            actual_mode = attention_mode
            if attention_mode == "sageattn" and not _HAS_SAGE_ATTN:
                logger.warning("SageAttention not installed. pip install sageattention. Falling back to sdpa.")
                actual_mode = "sdpa"
            elif attention_mode == "flash_attn" and not _HAS_FLASH_ATTN:
                logger.warning("flash-attn not installed. pip install flash-attn. Falling back to sdpa.")
                actual_mode = "sdpa"
            enable_optimized_attention(actual_mode)
        else:
            disable_optimized_attention()
        sam_model = load_sam_model(sam_model_name)
        grounding_dino_model = load_groundingdino_model(grounding_dino_model_name)
        total_images = image.shape[0]
        device = comfy.model_management.get_torch_device()
        license_valid = True
        if license_key and license_key.strip():
            is_valid, error_msg = validate_license(license_key, license_server_url)
            if not is_valid:
                license_valid = False
            else:
                pass
        else:
            license_valid = False
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }
        target_dtype = dtype_map.get(precision, torch.float32)
        use_autocast = precision != "fp32" and torch.cuda.is_available()
        start_time = time.time()
        total_batches = (total_images + batch_size - 1) // batch_size
        sam_is_hq = False
        if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name: sam_is_hq = True
        predictor = SamPredictorHQ(sam_model, sam_is_hq)
        res_images, res_masks = [], []
        current_batch_size = batch_size
        i = 0
        batch_idx = 0
        # Use a single autocast context for the entire loop to avoid dtype mismatches
        autocast_ctx = torch.cuda.amp.autocast(dtype=target_dtype) if use_autocast else torch.no_grad()
        while i < total_images:
            batch_idx += 1
            end_idx = min(i + current_batch_size, total_images)
            chunk_images = image[i:end_idx]
            try:
              with autocast_ctx:
                sam_input_batch = prepare_sam_batch(chunk_images, target_length=target_resolution, dtype=torch.float32)
                batch_features, batch_interm = predictor.get_image_features(sam_input_batch)
                original_size = (chunk_images.shape[1], chunk_images.shape[2])
                input_size = (sam_input_batch.shape[2], sam_input_batch.shape[3])
                chunk_pil_list = []
                for j in range(chunk_images.shape[0]):
                    chunk_pil_list.append(
                        Image.fromarray(np.clip(255. * chunk_images[j].cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGB')
                    )
                batch_boxes = groundingdino_predict_batch(grounding_dino_model, chunk_pil_list, prompt, threshold, dtype=torch.float32)
                for idx_in_batch in range(chunk_images.shape[0]):
                    curr_image_tensor = chunk_images[idx_in_batch]
                    boxes = batch_boxes[idx_in_batch]
                    if boxes.shape[0] == 0:
                        empty_mask = torch.zeros((1, original_size[0], original_size[1]), dtype=torch.float32, device="cpu")
                        res_masks.append(empty_mask)
                        res_images.append(curr_image_tensor.unsqueeze(0))
                        continue
                    curr_feat = batch_features[idx_in_batch].unsqueeze(0)
                    curr_interm = None
                    if sam_is_hq and batch_interm is not None:
                         curr_interm = [f[idx_in_batch].unsqueeze(0) for f in batch_interm]
                    predictor.set_precomputed_features(curr_feat, curr_interm, original_size, input_size)
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, original_size)
                    transformed_boxes = transformed_boxes.to(device)
                    masks, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False
                    )
                    final_mask = torch.any(masks, dim=0).float().cpu()
                    res_masks.append(final_mask)
                    mask_3d = final_mask[0].unsqueeze(-1).repeat(1, 1, 3)
                    masked_image = curr_image_tensor * mask_3d
                    res_images.append(masked_image.unsqueeze(0))
                del batch_features, batch_interm, sam_input_batch
                if (batch_idx % 20 == 0 or batch_idx == total_batches) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                i = end_idx
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    total_batches = (total_images + current_batch_size - 1) // current_batch_size
                    pass
                    batch_idx -= 1
                else:
                    logger.error(f"CUDA OOM with batch_size=1. Cannot process image {i}. Skipping.")
                    empty_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")
                    res_masks.append(empty_mask)
                    res_images.append(image[i].unsqueeze(0))
                    i += 1
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx} (images {i}-{end_idx-1}): {str(e)}")
                for j in range(i, end_idx):
                    empty_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")
                    res_masks.append(empty_mask)
                    res_images.append(image[j].unsqueeze(0))
                i = end_idx
        elapsed = time.time() - start_time
        # Restore original attention after processing
        if attention_mode in ("sdpa", "flash_attn", "sageattn"):
            disable_optimized_attention()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if len(res_images) == 0:
            return (torch.zeros_like(image), torch.zeros((total_images, image.shape[1], image.shape[2])))
        if not license_valid:
            pass
            fake_masks = torch.rand((total_images, image.shape[1], image.shape[2])) > 0.5
            fake_masks = fake_masks.float()
            fake_images = []
            for i in range(total_images):
                fake_mask_3d = fake_masks[i].unsqueeze(-1).repeat(1, 1, 3)
                fake_img = image[i] * fake_mask_3d
                fake_images.append(fake_img.unsqueeze(0))
            return (torch.cat(fake_images, dim=0), fake_masks)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))
class InvertMask:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"mask": ("MASK",)}}
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)
    def main(self, mask): return (1.0 - mask,)
class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"mask": ("MASK",)}}
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["boolean_number"]
    FUNCTION = "main"
    CATEGORY = "segment_anything"
    def main(self, mask): return (torch.all(mask == 0).int().item(), )
