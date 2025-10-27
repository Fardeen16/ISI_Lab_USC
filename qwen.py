import os
import csv
import argparse
import glob
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

# ---------------------------
# Helpers
# ---------------------------
def load_and_downscale(path: str, max_side: int = 1024) -> Image.Image:
    """Load an image and proportionally resize so the longest side <= max_side."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[INFO] Downscaled {os.path.basename(path)} {w}x{h} → {new_w}x{new_h}")
    return img

def pick_device_and_dtype():
    """Prefer CUDA. Use bf16 if supported; else fp16 on CUDA; else fp32."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ok else torch.float16
        print(f"[SETUP] GPU: {torch.cuda.get_device_name(0)} | dtype: {'bf16' if bf16_ok else 'fp16'}")
    else:
        dtype = torch.float32
        print("[SETUP] Using CPU fp32 (CUDA not available)")
    return device, dtype

def find_image_for_post(images_root: str, post_id: str):
    """
    Return a path to an image for a given post_id or None if not found.
    Tries common names then falls back to first image in the folder.
    """
    folder = os.path.join(images_root, post_id)
    if not os.path.isdir(folder):
        return None

    # 1) Try the common 'image_1' in typical extensions.
    candidates = [os.path.join(folder, f"image_1{ext}") for ext in (".jpg", ".jpeg", ".png", ".webp")]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # 2) Else take the first image-looking file.
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    return files[0] if files else None

# ---------------------------
# Core
# ---------------------------
def main():
    parser = argparse.ArgumentParser("Qwen Image Edit (GPU, VSCode)")

    # Batch mode (CSV) vs single image mode
    parser.add_argument("--csv", help="Path to CSV with columns postID, description")
    parser.add_argument("--images-root", default="images", help="Root folder containing images/<postID>/*")
    parser.add_argument("--outdir", default="qwen_edited", help="Folder to write edited images")
    parser.add_argument("--resume", action="store_true", help="Skip rows whose output already exists")

    # Single image fallback (kept for convenience)
    parser.add_argument("--image", help="Path to a single input image")
    parser.add_argument("--prompt", help="Edit instruction for single image")
    parser.add_argument("--out", default="output_image_edit.png", help="Output path for single image")

    # Inference knobs
    parser.add_argument("--steps", type=int, default=50, help="num_inference_steps")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="true_cfg_scale")
    parser.add_argument("--max-side", type=int, default=1024, help="Resize longest side to this (keeps aspect)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--neg", default=" ", help="negative_prompt")
    args = parser.parse_args()

    # Device / dtype
    device, dtype = pick_device_and_dtype()

    # Avoid CUDA fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

    # Load pipeline once and reuse
    print("[INFO] Loading Qwen/Qwen-Image-Edit …")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.float16,     # keep memory low
        low_cpu_mem_usage=True,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    try:
        pipe.enable_sequential_cpu_offload()  # offload inactive parts to CPU RAM
        print("[INFO] Using sequential CPU offload")
    except Exception as e:
        print("[WARN] Could not enable CPU offload:", e)

    # Generator on the correct device
    try:
        generator = torch.Generator(device="cuda" if device == "cuda" else "cpu").manual_seed(int(args.seed))
    except Exception:
        generator = torch.manual_seed(int(args.seed))

    # ---------- Single image path ----------
    if args.csv is None:
        if not args.image or not args.prompt:
            raise SystemExit("When --csv is not provided, you must pass --image and --prompt.")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        image = load_and_downscale(args.image, max_side=args.max_side)
        inputs = {
            "image": image,
            "prompt": args.prompt,
            "generator": generator,
            "true_cfg_scale": float(args.true_cfg_scale),
            "negative_prompt": args.neg,
            "num_inference_steps": int(args.steps),
        }
        print("[INFO] Running single-image inference …")
        if device == "cuda":
            compute_dtype = torch.bfloat16 if dtype is torch.bfloat16 else torch.float16
            with torch.autocast(device_type="cuda", dtype=compute_dtype), torch.inference_mode():
                out = pipe(**inputs)
        else:
            with torch.inference_mode():
                out = pipe(**inputs)
        out.images[0].save(args.out)
        print("✅ Saved:", os.path.abspath(args.out))
        return

    # ---------- Batch CSV path ----------
    # Read CSV safely (no pandas dependency)
    csv_path = args.csv
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    processed, skipped_missing, skipped_existing, failed = 0, 0, 0, 0

    # Support common header cases with different letter cases/whitespace
    def norm(s): return s.strip().lower()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {norm(h): h for h in reader.fieldnames or []}
        # Map normalized names to actual header names
        pid_col = headers.get("postid") or headers.get("post_id")
        desc_col = headers.get("output")
        if not pid_col or not desc_col:
            raise SystemExit("CSV must contain columns 'postID' and 'output' (case-insensitive).")

        for row in reader:
            post_id = (row.get(pid_col) or "").strip()
            prompt = (row.get(desc_col) or "").strip()
            if not post_id or not prompt:
                continue

            # Output path (always .png)
            out_path = os.path.join(outdir, f"{post_id}.png")
            if args.resume and os.path.exists(out_path):
                skipped_existing += 1
                continue

            # Find image for this post
            img_path = find_image_for_post(args.images_root, post_id)
            if img_path is None:
                print(f"[SKIP] No image for postID={post_id}")
                skipped_missing += 1
                continue

            try:
                image = load_and_downscale(img_path, max_side=args.max_side)
                inputs = {
                    "image": image,
                    "prompt": prompt,
                    "generator": generator,
                    "true_cfg_scale": float(args.true_cfg_scale),
                    "negative_prompt": args.neg,
                    "num_inference_steps": int(args.steps),
                }

                # Inference
                if device == "cuda":
                    compute_dtype = torch.bfloat16 if dtype is torch.bfloat16 else torch.float16
                    with torch.autocast(device_type="cuda", dtype=compute_dtype), torch.inference_mode():
                        out = pipe(**inputs)
                else:
                    with torch.inference_mode():
                        out = pipe(**inputs)

                out.images[0].save(out_path)
                processed += 1
                print(f"✅ Saved: {out_path}")
            except Exception as e:
                failed += 1
                print(f"[ERROR] postID={post_id} img={img_path}: {e}")

    print(f"\n[SUMMARY] processed={processed}  skipped_missing={skipped_missing}  skipped_existing={skipped_existing}  failed={failed}")

if __name__ == "__main__":
    main()



# import os
# import argparse
# import torch
# from PIL import Image
# from diffusers import QwenImageEditPipeline


# # ---------------------------
# # Helpers
# # ---------------------------
# def load_and_downscale(path: str, max_side: int = 1024) -> Image.Image:
#     """
#     Load an image and proportionally resize it so that the longest side <= max_side.
#     Preserves aspect ratio. Returns a PIL.Image in RGB mode.
#     """
#     img = Image.open(path).convert("RGB")
#     w, h = img.size
#     longest = max(w, h)
#     if longest > max_side:
#         scale = max_side / longest
#         new_w, new_h = int(w * scale), int(h * scale)
#         img = img.resize((new_w, new_h), Image.LANCZOS)
#         print(f"[INFO] Downscaled {os.path.basename(path)} {w}x{h} → {new_w}x{new_h}")
#     return img


# def pick_device_and_dtype():
#     """
#     Prefer CUDA. Use bf16 if supported; else fp16 on CUDA; else fp32.
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if device == "cuda":
#         bf16_ok = torch.cuda.is_bf16_supported()
#         dtype = torch.bfloat16 if bf16_ok else torch.float16
#         print(f"[SETUP] GPU: {torch.cuda.get_device_name(0)} | dtype: {'bf16' if bf16_ok else 'fp16'}")
#     else:
#         dtype = torch.float32
#         print("[SETUP] Using CPU fp32 (CUDA not available)")
#     return device, dtype


# # ---------------------------
# # Core
# # ---------------------------
# def main():
#     parser = argparse.ArgumentParser("Qwen Image Edit (GPU, VSCode)")
#     parser.add_argument("--image", required=True, help="Path to input image")
#     parser.add_argument("--prompt", required=True, help="Edit instruction")
#     parser.add_argument("--out", default="output_image_edit.png", help="Output path")
#     parser.add_argument("--steps", type=int, default=50, help="num_inference_steps")
#     parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="true_cfg_scale")
#     parser.add_argument("--max-side", type=int, default=1024, help="Resize longest side to this (keeps aspect)")
#     parser.add_argument("--seed", type=int, default=0, help="Random seed")
#     parser.add_argument("--neg", default=" ", help="negative_prompt")
#     args = parser.parse_args()

#     # Device / dtype
#     device, dtype = pick_device_and_dtype()

#     # Optional: helps CUDA memory fragmentation on Windows
#     os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

#     # Load pipeline
#     print("[INFO] Loading Qwen/Qwen-Image-Edit …")
#     pipe = QwenImageEditPipeline.from_pretrained(
#         "Qwen/Qwen-Image-Edit",
#         torch_dtype=torch.float16,     # keep memory low
#         low_cpu_mem_usage=True,
#         )
    
#     pipe.enable_attention_slicing()
#     pipe.enable_vae_tiling()

#     # Move to device/dtype safely with fallbacks
#     try:
#         # pipe.to(dtype)
#         # pipe.to(device)
#         pipe.enable_sequential_cpu_offload()
#     except Exception as e:
#         print(f"[WARN] pipeline.to({device}, {dtype}) failed: {e}")
#         if device == "cuda" and dtype is torch.bfloat16:
#             print("[INFO] Retrying with float16 on CUDA …")
#             pipe.to(torch.float16).to("cuda")
#             dtype = torch.float16
#         else:
#             print("[INFO] Falling back to CPU fp32 …")
#             pipe.to(torch.float32).to("cpu")
#             device, dtype = "cpu", torch.float32

#     # VRAM helpers
#     #pipe.enable_vae_tiling()
#     # If tight on VRAM, uncomment one or both:
#     # pipe.enable_attention_slicing()
#     # pipe.enable_sequential_cpu_offload()

#     # Prepare input image
#     image = load_and_downscale(args.image, max_side=args.max_side)

#     # Generator on the correct device
#     try:
#         generator = torch.Generator(device=device).manual_seed(int(args.seed))
#     except Exception:
#         generator = torch.manual_seed(int(args.seed))

#     inputs = {
#         "image": image,
#         "prompt": args.prompt,
#         "generator": generator,
#         "true_cfg_scale": float(args.true_cfg_scale),
#         "negative_prompt": args.neg,
#         "num_inference_steps": int(args.steps),
#     }

#     # Inference (autocast on CUDA)
#     print("[INFO] Running inference …")
#     if device == "cuda":
#         compute_dtype = torch.bfloat16 if dtype is torch.bfloat16 else torch.float16
#         with torch.autocast(device_type="cuda", dtype=compute_dtype), torch.inference_mode():
#             out = pipe(**inputs)
#     else:
#         with torch.inference_mode():
#             out = pipe(**inputs)

#     img = out.images[0]
#     img.save(args.out)
#     print("✅ Saved:", os.path.abspath(args.out))


# if __name__ == "__main__":
#     main()

