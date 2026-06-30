import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
import time


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
	"""Aplica el filtro CLAHE para mejorar el contraste."""
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
	return clahe.apply(image)

def apply_top_hat(image, kernel_size=(15, 15)):
	"""Aplica el filtro Top-Hat para resaltar estructuras brillantes pequeñas."""
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
	# return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
	new_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
	# _, ret_image = cv2.threshold(new_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# ret_image = cv2.equalizeHist(new_image)
	# new_image = apply_clahe(new_image)
	# ret_image = np.clip(image.astype(np.uint32)+new_image.astype(np.uint32), 0, 255).astype(np.uint8)
	return new_image

def apply_morph_close(image, kernel_size=(15, 15)):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)	
	new_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	return new_image

def apply_entropy(image):

	# mean = cv2.blur(image, (5,5))
	# sqr_mean = cv2.blur(image**2, (5,5))
	# variance = sqr_mean -mean**2
	# new_image = cv2.normalize(255/(variance+1), None, 0, 255, cv2.NORM_MINMAX)

	image = apply_clahe(image)
	# image = apply_morph_close(image, kernel_size=(15,15))
	# ent = entropy(image, disk(15))
	# # new_image = image - cv2.normalize(ent, None, 0, 1, cv2.NORM_MINMAX)*image 
	# new_image = (1 - ent/ent.max())*image 
	# # new_image = new_image#.astype(np.uint8)

	# new_image = cv2.medianBlur(image, 5)
	gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
	gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
	grad = cv2.magnitude(gx, gy)
	k=9
	gmean = cv2.boxFilter(grad, -1, (k,k), normalize=True)
	new_image = (1-gmean/gmean.max())*image
	# new_image = image-gmean #cv2.normalize(1.0/(gmean+1e-3), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	# new_image = cv2.normalize(gmean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	return new_image



def gabor_bank(image, ksize=(31, 31), sigma=5.0, gamma=0.5, psi=0, lambdas=None, thetas=None):
	"""Aplica un banco de filtros de Gabor y devuelve la imagen de máxima respuesta."""
	if lambdas is None:
		lambdas = [10.0, 20.0, 30.0]
	if thetas is None:
		thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
	gabor_responses = []
	for lmbda in lambdas:
		for theta in thetas:
			gabor_kernel = cv2.getGaborKernel(ksize, sigma, theta, lmbda, gamma, psi, ktype=cv2.CV_32F)
			filtered_img = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
			gabor_responses.append(np.abs(filtered_img))
	max_response_image = np.max(np.array(gabor_responses), axis=0)
	return max_response_image

def binarize_and_morph(image, threshold_value=205, close_kernel_size=(15, 15), open_kernel_size=(7, 7), close_iter=3, open_iter=2):
	"""Binariza y aplica operaciones morfológicas (cierre y apertura) para segmentación."""
	gabor_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	_, binary_mask = cv2.threshold(gabor_normalized, threshold_value, 255, cv2.THRESH_BINARY)
	close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
	closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iter)
	open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
	final_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel, iterations=open_iter)
	return final_mask

import cv2

def resize_width_and_rois(image, labels, factor, interpolation=None):
    """ Interpolation: None => linear si factor>=1, area si factor<1"""
    if image is None or image.size == 0:
        raise ValueError("image vacía o None")
    if factor <= 0:
        raise ValueError("factor debe ser > 0")

    H, W = image.shape[:2]
    new_W = max(1, int(round(W * factor)))
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR if factor >= 1.0 else cv2.INTER_AREA

    new_image = cv2.resize(image, (new_W, H), interpolation=interpolation)

    def _adjust_list(boxes):
        out = []
        for b in boxes:
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            x, y, w, h = map(float, b)
            x_scaled = x * factor
            w_scaled = w * factor
            # recorte horizontal al rango [0, new_W)
            x_end = x_scaled + w_scaled
            inter_left  = max(0.0, x_scaled)
            inter_right = min(float(new_W), x_end)
            inter_w = inter_right - inter_left
            if inter_w <= 0:
                continue
            out.append([inter_left, y, inter_w, h])
        return out

    if isinstance(labels, dict):
        new_labels = {}
        for cls_name, boxes in labels.items():
            new_labels[cls_name] = _adjust_list(boxes) if isinstance(boxes, list) else boxes
    elif isinstance(labels, list):
        new_labels = _adjust_list(labels)
    else:
        new_labels = labels

    return new_image, new_labels
