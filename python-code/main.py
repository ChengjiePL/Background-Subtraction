import os

from PIL.Image import init
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import morphology

# TASCA 1

def load_dataset():
    dataset = '../highway/input/'
    files = sorted([f for f in os.listdir(dataset) if f.startswith("in001") and f.endswith(".jpg")])
    train_files = files[:150]
    test_files = files[150:]
    print("✔ Successfully load Dataset.")
    return dataset, train_files, test_files

# TASCA 2

def calculate_mean_std(dataset, test_files, train_files):
    # Cargar imágenes y apilarlas en un array 3D (N x H x W)
    images = np.stack([io.imread(os.path.join(dataset, fname), as_gray=True) for fname in train_files])
    test_im = io.imread(os.path.join(dataset, test_files[0]), as_gray=True).astype(float)

    # Ahora sí podemos usar np.mean() y np.std() correctamente
    mean_bg = np.mean(images, axis=0)  # Media píxel a píxel
    std_bg = np.std(images, axis=0)    # Desviación estándar píxel a píxel

    print(f'✔ Successfully calculate Mean and Standard Deviation. {mean_bg.shape, std_bg.shape}')

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(mean_bg, cmap="gray")
    plt.title("Mitjana del fons")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Desviación estandar")
    plt.imshow(std_bg, cmap="gray")
    plt.axis('off')

    plt.show()

    return mean_bg, std_bg


# TASCA 3
def segmentar_coches_mean(dataset, test_files, mean_bg):
    thr = 0.3
    test_im = io.imread(os.path.join(dataset, test_files[0]), as_gray=True).astype(np.float64)
    fg_mask = np.abs(test_im - mean_bg) > thr

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(test_im, cmap="gray"); plt.title("Imatge original")
    plt.subplot(1,2,2); plt.imshow(fg_mask, cmap="gray"); plt.title("Mascara de foreground")
    plt.show()
    return test_im


# TASCA 4
def segmentar_coches_std(dataset, test_files, mean_bg, std_bg, test_im):
    alpha = 0.6
    beta = 0.2

    # Nueva segmentación con desviación estándar
    fg_mask = np.abs(test_im - mean_bg) > (alpha * std_bg + beta)

    plt.imshow(fg_mask, cmap="gray")
    plt.title("Segmentación mejorada con modelo refinado")
    plt.show()

    return alpha, beta


# TASCA 5
def video_resultat(dataset, test_files, mean_bg, alpha,  beta, std_bg, test_im):
    out = cv2.VideoWriter('resultat.avi', cv2.VideoWriter_fourcc(*'mp4v'), 10, (test_im.shape[1], test_im.shape[0]), False)

    for fname in test_files:
        test_im = io.imread(os.path.join(dataset, fname), as_gray=True).astype(np.float64)
        fg_mask = np.abs(test_im - mean_bg) > (alpha * std_bg + beta)
        out.write((fg_mask * 255).astype(np.uint8))

    out.release()
    print("🎥 Vídeo guardado como resultat.mp4")


# TASCA 6
def evaluate_performance(dataset, test_files, mean_bg, std_bg):
    groundtruth_path = '../highway/groundtruth/'
    
    # Define evaluation cases (thresholds scaled to [0,1] range)
    cases = [
        {'name': 'Simple Model (thr=0.27)', 'thr': 70/255, 'alpha': None, 'beta': None},
        {'name': 'Elaborate Model (α=0.45, β=0.06)', 'thr': None, 'alpha': 0.45, 'beta': 15/255},
        {'name': 'Elaborate Model (α=0.6, β=0.08)', 'thr': None, 'alpha': 0.6, 'beta': 20/255}
    ]
    
    metrics = [{'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for _ in cases]

    for test_file in test_files:
        # Load test image
        test_im = io.imread(os.path.join(dataset, test_file), as_gray=True).astype(float)
        
        # Load ground truth (convert to 2D and binarize)
        base_name = test_file.replace("in", "gt").replace(".jpg", ".png")
        gt = io.imread(os.path.join(groundtruth_path, base_name), as_gray=True)
        gt_binary = (gt > 0.5).astype(bool)  # Assuming ground truth is [0,1] with 1=foreground

        for i, case in enumerate(cases):
            # Generate segmentation mask
            if case['thr'] is not None:
                mask = np.abs(test_im - mean_bg) > case['thr']
            else:
                threshold = case['alpha'] * std_bg + case['beta']
                mask = np.abs(test_im - mean_bg) > threshold
                # Post-processing
                mask = morphology.binary_closing(mask, morphology.disk(3))
                mask = morphology.remove_small_objects(mask, min_size=50)
                mask = morphology.remove_small_holes(mask, area_threshold=50)

            # Calculate metrics
            tp = np.sum(mask & gt_binary)
            fp = np.sum(mask & ~gt_binary)
            tn = np.sum(~mask & ~gt_binary)
            fn = np.sum(~mask & gt_binary)

            # Update metrics
            metrics[i]['tp'] += tp
            metrics[i]['fp'] += fp
            metrics[i]['tn'] += tn
            metrics[i]['fn'] += fn

    # Print results
    print("\nTASCA 6 - Performance Evaluation")
    for i, case in enumerate(cases):
        tp = metrics[i]['tp']
        fp = metrics[i]['fp']
        tn = metrics[i]['tn']
        fn = metrics[i]['fn']
        
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nCase {i+1}: {case['name']}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    return metrics

def main():
    # TASCA 1
    dataset, train_files, test_files = load_dataset()

    # TASCA 2
    mean_bg, std_bg = calculate_mean_std(dataset, test_files, train_files)

    # TASCA 3
    test_im = segmentar_coches_mean(dataset, test_files, mean_bg)

    # TASCA 4
    alpha, beta = segmentar_coches_std(dataset, test_files, mean_bg, std_bg, test_im)
    # TASCA 5
    video_resultat(dataset, test_files, mean_bg, alpha, beta, std_bg, test_im)

    # TASCA 6
    evaluate_performance(dataset, test_files, mean_bg, std_bg)

if __name__ == "__main__":
    main()
