"""All necessary image operations for the SMART algorithm
"""

import numpy as np
from math import tan, pi

import cv2

import params


def cosine_corrected(img, center, disk_radius):
    """ Applies a cosine correction to all pixels. The correction is applied to
    all pixel with an angle smaller than 60 degrees to the view axis. Otherwise
    there would be extrem high correction factors.
    """

    # create coordinate grid, where 'center' is the origin of the coordinate
    # aka the center of the sun
    x, y = np.mgrid[-center[0]:-center[0] + img.shape[0],
                    -center[1]:-center[1] + img.shape[1]]
    dist = np.sqrt(x * x + y * y)

    max_correction_angle = tan(pi / 3)

    cos_correction = np.sin(np.arccos(dist / disk_radius))
    cos_correction[cos_correction < max_correction_angle] = 1
    corrected = img / cos_correction

    return np.nan_to_num(corrected)


def binarize(img):
    """Threshold image by its absolute values.
    Returns an image where all values between -range_ and range_ are set to
    zero and all other values are set to one.
    """
    return (~cv2.inRange(img, -params.STATIC_BACKGROUND_THRESHOLD,
                         params.STATIC_BACKGROUND_THRESHOLD)) & 1


def process_stl(img, center, disk_radius):
    """Apply smoothing, thresholding and LOS-correction to an LOS-magnetogram.
    LOS-correction with cosine: currently not in use, but implemented
    """
    ret = cv2.GaussianBlur(img, (
        params.GAUSSIAN_BLUR_KERNEL_SIZE,
        params.GAUSSIAN_BLUR_KERNEL_SIZE),
        params.GAUSSIAN_BLUR_SIGMA)

    # treshold noise values. -> is this necessary? will binarize later anyway
    # with the same parameters
    ret *= (~cv2.inRange(ret, -params.STATIC_BACKGROUND_THRESHOLD,
                         params.STATIC_BACKGROUND_THRESHOLD)) & 1

    # cosine correction
    ret = cosine_corrected(ret, center, disk_radius)

    return ret


def create_circle_mask(radius, center, shape):
    xx, yy = np.mgrid[-center[1]:-center[1] + shape[1],
                      -center[0]:-center[0] + shape[0]]
    circle = (xx ** 2 + yy ** 2) < radius * radius

    return circle.astype(np.uint8)


def extract_features(hmi_t, hmi_dt, center, disk_radius):
    """Extracts the contours of all features on a LOS-magnetogram.

    HMI_t: The HMI-magnetogram (line-of-sight) to examine.
    HMI_t_delta: A previous HMI-magnetogram, used to extract time-dependent
      features.
    center: The center of the sun disk in pixels on the image.
    delta_time: The time between HMI_t and HMI_t_delta.
    disk_radius: the radius of the sun in pixels.
    """

    m_t = process_stl(hmi_t, center, disk_radius)
    m_t = binarize(m_t)

    m_t_delta = process_stl(hmi_dt, center, disk_radius)
    m_t_delta = binarize(m_t_delta)

    r = params.FEATURE_DILATION_RADIUS
    kernel = create_circle_mask(r, (r, r), (2 * r, 2 * r))
    m_t_grown = cv2.dilate(m_t, kernel)
    m_t_delta_grown = cv2.dilate(m_t_delta, kernel)
    diff = cv2.bitwise_xor(m_t_grown, m_t_delta_grown)

    igm_t = cv2.subtract(m_t, diff)
    igm_t = cv2.dilate(igm_t, kernel)

    contours = cv2.findContours(
        igm_t, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )[0]

    ret = []
    for i in contours:
        shape = cv2.boundingRect(i)
        if shape[2] * shape[3] >= params.MINIMAL_FEATURE_AREA:
                ret.append(i)

    return ret
