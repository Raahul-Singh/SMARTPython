from datetime import datetime

# the border of the sun disk contains some artefacts, so reduce the radius
USABLE_DISK_RADIUS = 0.95  # factor
STATIC_BACKGROUND_THRESHOLD = 70  # Gauss
MINIMAL_FEATURE_AREA = 200  # pixel
GAUSSIAN_BLUR_KERNEL_SIZE = 41  # pixel
GAUSSIAN_BLUR_SIGMA = 8
FEATURE_DILATION_RADIUS = 40  # pixel
PSL_DILATION_RADIUS = 16  # pixel, is 4096 / 250 => same ratio as for MDI
SUN_DIAMETER_M = 1392684000.0  # diameter of sun in meters
SUN_DIAMETER_P = 3718.0  # diameter of sun in pixels (on an HMI image)
METERS_PER_PIXEL = SUN_DIAMETER_M / SUN_DIAMETER_P  # in meters
AREA_PER_PIXEL = METERS_PER_PIXEL ** 2  # m^2/pixel at disk center
SG_THRESHOLD = 50.0

# time range to process
START = datetime(2014, 1, 1, 0)
END = datetime(2014, 1, 1, 1)
