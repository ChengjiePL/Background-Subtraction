import os

from PIL.Image import init
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim

# TASCA 1

def load_dataset():
    dataset = './highway/input/'
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
def avaluar_model(test_files, dataset, mean_bg, alpha, beta, std_bg):
    # Cargar groundtruth
    groundtruth_path = "highway/groundtruth/"
    gt_files = sorted([f for f in os.listdir(groundtruth_path) if f.startswith("gt001") and f.endswith(".png")])

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i, fname in enumerate(test_files):
        test_im = io.imread(os.path.join(dataset, fname), as_gray=True).astype(np.float64)
        fg_mask = np.abs(test_im - mean_bg) > (alpha * std_bg + beta)
        
        gt = io.imread(os.path.join(groundtruth_path, gt_files[i]), as_gray=True) > 0  # Convertir a binario
        
        tp = np.sum(fg_mask * gt)
        fp = np.sum(fg_mask) - tp
        fn = np.sum(gt) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = np.sum(fg_mask == gt) / gt.size
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        accuracies.append(accuracy)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    mean_accuracies = np.mean(accuracies)

    print(f"📊 Precision: {mean_precision:.3f}, Recall: {mean_recall:.3f}, F1-score: {mean_f1:.3f}, accuracy: {mean_accuracies:.3f}")

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
    avaluar_model(test_files, dataset, mean_bg, alpha, beta, std_bg)

if __name__ == "__main__":
    main()
