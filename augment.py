# Copyright (c) 2020 smarsu. All Rights Reserved.

"""Data augmentation for images.

TODO(smarsu): Data augmentation for text and voice.
"""

import cv2
import numpy as np


def augment(imgs, bboxes=None):
  """Do data augmentation for imgs.
  
  Args:
    imgs: ndarray. [n, h, w, 3]
    bboxes: ndarray. [n, ?, 4]

  Returns:
    imgs: ndarray. [n, h, w, 3]
    bboxes: ndarray. [n, ?, 4]
  """
  if bboxes is None:
    imgs = soft_augment(imgs)
  else:
    pass

  return imgs, bboxes


def soft_augment(imgs):
  """Data augmentation without change the location of pixels.
  
  Args:
    imgs: ndarray. shape [n, h, w, 3]

  Returns:
    imgs: ndarray. shape [n, h, w, 3]
  """
  ret = []
  for img in imgs:
    img = blur(img)
    img = scale(img)
    img = dropout(img)
    ret.append(img)
  return np.concatenate(ret, 0)


def blur(img, scale=0.35):
  """Blur the image by resize to small image then resize back.

  Args:
    img: ndarray. shape [h, w, 3].
    scale: float. 

  Returns:
    img: ndarray. shape [h, w, 3]. The blurred image.
  """
  if scale >= 1:
    return img

  scale = np.random.uniform(scale, 1)
  h, w, _ = img.shape
  img = cv2.resize(img, (round(w * scale), round(h * scale)))
  img = cv2.resize(img, (w, h))
  return img


def scale(img, alpha=(0.9, 10/9), beta=(-5, 5)):
  """Change brightness, saturation and contrast of the image.
  
  img = alpha * img + beta.

  Args:
    img: ndarray. shape [h, w, 3].
    alpha: 
    beta:

  Returns:
    img: ndarray. shape [h, w, 3]
  """
  alpha = np.random.uniform(alpha[0], alpha[1])
  beta = np.random.uniform(beta[0], beta[1])
  img = alpha * img + beta
  # Avoid the pixel overflow [0, 255], which is the uint8_t limit.
  img = np.maximum(img, 0)
  img = np.minimum(img, 255)
  return img


def dropout(img, scale=0.99):
  """Random dropout some pixel in the image.
  
  Args:
    img: ndarray. shape [h, w, 3]
    scale: float

  Returns:
    img: ndarray. shape [h, w, 3]
  """
  size = round(img.size * scale)
  inv_size = img.size - size
  keep = np.concatenate([np.ones(size), np.zeros(inv_size)], 0)
  np.random.shuffle(keep)
  keep = keep.reshape(img.shape)
  inv_keep = np.logical_not(keep)
  mask = np.random.randint(0, 256, img.shape)
  return keep * img + inv_keep * mask


if __name__ == '__main__':
  test_blur = True
  test_scale = True
  test_dropout = True
  test_all = True

  src_img = cv2.imread('test.jpg')
  cv2.imshow('src', src_img)

  if test_blur:
    print('Test Blur')
    img = blur(src_img).astype(np.uint8)
    cv2.imshow('blur', img)

  if test_scale:
    print('Test Scale')
    img = scale(src_img).astype(np.uint8)
    cv2.imshow('scale', img)

  if test_dropout:
    print('Test Dropout')
    img = dropout(src_img).astype(np.uint8)
    cv2.imshow('dropout', img)

  if test_all:
    print('Test All')
    img = blur(src_img).astype(np.uint8)
    img = scale(img).astype(np.uint8)
    img = dropout(img).astype(np.uint8)
    cv2.imshow('all', img)

  cv2.waitKey(0)
