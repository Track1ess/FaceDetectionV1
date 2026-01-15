#!/usr/bin/env python3
import argparse, csv, time, sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from facenet_pytorch import MTCNN

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS: yield p

def to_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    return im.convert("RGB") if im.mode != "RGB" else im

def sync_cuda():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def percentiles(vals, qs=(50,90)):
    if not vals: return {q: float("nan") for q in qs}
    a = np.asarray(vals, np.float64)
    return {q: float(np.percentile(a, q)) for q in qs}

def load_mobilefacenet(repo_root: Path, weights: str, device):
    import sys
    sys.path.append(str(repo_root))                 # repo root contains mobilefacenet.py
    from mobilefacenet import MobileFaceNet         # foamliu's class

    model = MobileFaceNet().to(device).eval()       # <-- no embedding_size here

    sd = torch.load(weights, map_location="cpu")
    # handle different checkpoint formats
    for key in ("state_dict", "model_state_dict", "net_state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    # strip "module." if it exists (DDP)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="/home/ntu/CLionProjects/faceDetection/build", help="Folder with images (recursively scanned)")
    ap.add_argument("--repo_root", default="/home/ntu/Documents/face_recognition/mobilefacenet/MobileFaceNet", help="Path to foamliu/MobileFaceNet")
    ap.add_argument("--weights", default="/home/ntu/Documents/face_recognition/mobilefacenet/MobileFaceNet/retinaface/weights/mobilenet0.25_Final.pth")
    args = ap.parse_args()

    folder = Path(args.folder); assert folder.is_dir()
    repo_root = Path(args.repo_root); assert repo_root.exists()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Detect & align to 112x112 (InsightFace/MobileFaceNet standard)
    mtcnn = MTCNN(image_size=112, margin=20, keep_all=False, device=device)

    model = load_mobilefacenet(repo_root, args.weights, device)

    filenames, probs = [], []
    detect_ms_list, embed_ms_list, total_ms_list = [], [], []
    detected_flags, embeddings = [], []

    t0_all = time.perf_counter()
    with torch.no_grad():
        for path in list_images(folder):
            filenames.append(str(path))
            img = to_rgb(path)

            t0 = time.perf_counter()
            face, prob = mtcnn(img, return_prob=True)   # [3,112,112] or None
            sync_cuda()
            t1 = time.perf_counter()

            det = face is not None
            detected_flags.append(det)
            probs.append(float(prob) if det else 0.0)

            if not det:
                detect_ms_list.append((t1 - t0) * 1000.0)
                embed_ms_list.append(float("nan"))
                total_ms_list.append((t1 - t0) * 1000.0)
                continue

            if face.ndim == 3: face = face.unsqueeze(0)     # (1,3,112,112)
            face = (face * 2.0 - 1.0).to(device)            # [-1,1]

            e0 = time.perf_counter()
            emb = model(face)                                # (1,512)
            emb = F.normalize(emb, dim=1)
            sync_cuda()
            e1 = time.perf_counter()

            embeddings.append(emb[0].cpu().numpy().astype(np.float32))

            detect_ms_list.append((t1 - t0) * 1000.0)
            embed_ms_list.append((e1 - e0) * 1000.0)
            total_ms_list.append((e1 - t0) * 1000.0)

    t1_all = time.perf_counter()
    elapsed = t1_all - t0_all

    # Save outputs
    E = np.stack(embeddings, axis=0).astype(np.float32) if embeddings else np.empty((0,512), np.float32)
    np.save("embeddings_mobile.npy", E)

    with open("results_mobile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","detected","prob","detect_ms","embed_ms","total_ms"])
        for i, fn in enumerate(filenames):
            w.writerow([
                fn, detected_flags[i], f"{probs[i]:.6f}",
                f"{detect_ms_list[i]:.3f}",
                "" if np.isnan(embed_ms_list[i]) else f"{embed_ms_list[i]:.3f}",
                f"{total_ms_list[i]:.3f}",
            ])

    # Summary
    n = len(filenames); n_det = sum(detected_flags)
    fps = n/elapsed if elapsed>0 else float("inf")
    p_det = percentiles([x for x in detect_ms_list if not np.isnan(x)])
    p_emb = percentiles([x for x in embed_ms_list if not np.isnan(x)])
    p_tot = percentiles([x for x in total_ms_list if not np.isnan(x)])

    print("\n===== Summary (foamliu MobileFaceNet) =====")
    print(f"Images scanned      : {n}")
    print(f"Faces detected      : {n_det}")
    print(f"Total elapsed (s)   : {elapsed:.3f}")
    print(f"Throughput (imgs/s) : {fps:.2f}")
    if n:
        print(f"Avg detect (ms)     : {np.nanmean(detect_ms_list):.3f}")
        print(f"Avg embed  (ms)     : {np.nanmean(embed_ms_list):.3f}")
        print(f"Avg total  (ms)     : {np.nanmean(total_ms_list):.3f}")
        print(f"p50 detect/embed/total (ms): {p_det[50]:.3f} / {p_emb[50]:.3f} / {p_tot[50]:.3f}")
        print(f"p90 detect/embed/total (ms): {p_det[90]:.3f} / {p_emb[90]:.3f} / {p_tot[90]:.3f}")

    print("\nSaved: embeddings_mobile.npy, results_mobile.csv")

if __name__ == "__main__":
    main()
