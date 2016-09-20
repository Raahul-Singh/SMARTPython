"""This class represents a single magnetogram with some metadata. There is a
method to extract feature/contours from the magnetogram. Magnetograms are
considered to be from JSOC, otherwise some FITS-keyword may not be found.
"""

import astropy.io.fits as fits
import numpy as np
from datetime import datetime

import sunpy.wcs.wcs as wcs

import cv2

import img_operations
import params

ERROR_MSG = "%s seems not to be an HMI-magnetogram level 1.5"
STRING_TO_DATETIME = "%Y.%m.%d_%H:%M:%S_TAI"


class HMIMagnetogram:
    def __init__(self, file_):
        f = fits.open(file_)
        f.verify("fix")

        assert f[1].header["TELESCOP"] == "SDO/HMI", ERROR_MSG % file_
        assert f[1].header["BUNIT"] == "Gauss", ERROR_MSG % file_
        assert f[1].header["CONTENT"] == "MAGNETOGRAM", ERROR_MSG % file_

        self.disk_center = (f[1].header["CRPIX1"], f[1].header["CRPIX2"])
        # the border of the sun disk contains some artefacts, so reduce radius
        rsun_obs = f[1].header["RSUN_OBS"]
        self.pixel_scale = f[1].header["CDELT1"]
        self.disk_radius = int(
            params.USABLE_DISK_RADIUS * rsun_obs / self.pixel_scale)

        self.data = np.nan_to_num(np.array(f[1].data, np.float32))

        # rotate and veritcal-flip data, to get the right view
        rotation = f[1].header["CROTA2"]
        rotation_matrix = cv2.getRotationMatrix2D(self.disk_center, rotation,
                                                  1)
        self.data = cv2.warpAffine(cv2.flip(self.data, 0), rotation_matrix,
                                   self.data.shape)

        # set the values outside of the sun to 0
        self.data_mask = img_operations.create_circle_mask(self.disk_radius,
                                                           self.disk_center,
                                                           self.data.shape)
        self.data *= self.data_mask

        self.header = f[1].header
        self.noise_level = params.STATIC_BACKGROUND_THRESHOLD
        self.minimal_feature_area = params.MINIMAL_FEATURE_AREA
        self.delta_magnetogram = None
        self.shape = self.data.shape

        self.helioprojective_coordinates = None
        self.heliocentric_coordinates = None
        self.heliographic_coordinates = None
        self.cosine_map = None

        self.time = datetime.strptime(self.header["T_REC"],
                                      STRING_TO_DATETIME)

    def as_image(self, size=None, contours=None):
        tmp = self.data
        if size:
            tmp = cv2.resize(self.data, size)
        tmp[tmp < -2048] = -2048
        tmp[tmp > 2047] = 2047
        tmp += 2048
        tmp *= cv2.resize(self.data_mask, size)
        tmp = np.right_shift(tmp.astype(np.uint32), 4)

        tmp = tmp.astype(np.uint8)

        if contours:
            cv2.drawContours(tmp, contours, -1, (255, 0, 0), 2)

        return tmp

    def get_contours(self, delta_magnetogram=None):
        """delta_magnetogram should be differential rotated"""
        if self.delta_magnetogram is None:
            delta_magnetogram = self
        contours = img_operations.extract_features(self.data,
                                                   delta_magnetogram.data,
                                                   self.disk_center,
                                                   self.disk_radius)

        return contours

    def get_helioprojective_coordinates(self):
        if self.helioprojective_coordinates is None:
            self.helioprojective_coordinates = wcs.convert_pixel_to_data(
                self.data.shape,
                [self.header["CDELT1"], -self.header["CDELT2"]],
                # negate because we flipped the data?
                [self.header["CRPIX1"], self.header["CRPIX2"]],
                [self.header["CRVAL1"], self.header["CRVAL2"]])
        return self.helioprojective_coordinates

    def get_heliocentric_coordinates(self):
        if self.heliocentric_coordinates is None:
            x_coords, y_coords = self.get_helioprojective_coordinates()
            self.heliocentric_coordinates = \
                wcs.convert_hpc_hcc(x_coords, y_coords,
                                    dsun_meters=self.header["DSUN_OBS"],
                                    z=True)
        return self.heliocentric_coordinates

    def get_cosine_map(self):
        if self.cosine_map is None:
            hcc_z = self.get_heliocentric_coordinates()[2]
            self.cosine_map = cv2.normalize(hcc_z, None, 0, 1, cv2.NORM_MINMAX)
        return self.cosine_map

    def get_heliographic_coordinates(self):
        if self.heliographic_coordinates is None:
            hcc_x, hcc_y, hcc_z = self.get_heliocentric_coordinates()
            self.heliographic_coordinates = wcs.convert_hcc_hg(hcc_x, hcc_y,
                                                               hcc_z)[0:2]

        return self.heliographic_coordinates
