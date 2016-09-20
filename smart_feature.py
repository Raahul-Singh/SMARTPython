"""
Represents one feature produced by the feature extraction algorithm of SMART.
"""
import numpy as np
import scipy.stats
import scipy
import iso8601
import json

import cv2

import img_operations
import params


class SMARTFeature:
    def __init__(self):
        self.id = None
        self.mask = None
        self.contour = None
        self.phi_imb = None
        self.WL_sg_star = None
        self.R_star = None
        self.SG_len = None
        self.PSL_len = None
        self.phi_net_emrg = None
        self.phi_imb = None
        self.phi_abs = None
        self.phi_neg = None
        self.phi_pos = None
        self.area = None
        self.kurtosis = None
        self.skewness = None
        self.variance = None
        self.mean = None
        self.abs_sum = None
        self.sum = None
        self.min = None
        self.max = None
        self.position = None
        self.time = None
        self.index = None
        self.shape = None

    @classmethod
    def from_hmi(cls, hmi_magnetogram, index, feature_contour, delta_t,
                 delta_t_magnetogram):
        """
        Parameters:
          hmi_magnetogram: HMIMagnetogram instance
          index: A number which describes the index of this feature on the
                 magnetogram.
          feature_contour: Set of pixels which describe feature area.
          delta_t: Time difference between this and last magnetogram,
          in seconds.
          delta_t_magnetogram: HMIMagnetogram to calculate delta phi,
          may be None
        """

        o = cls()

        # initial code
        o.index = index
        o.time = hmi_magnetogram.time
        x, y, w, h = cv2.boundingRect(feature_contour)
        o.shape = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        o.contour = feature_contour.squeeze()
        o.mask = np.zeros((h, w), dtype=np.uint8)

        cv2.drawContours(o.mask, [feature_contour - (x, y)], 0, 1, -1)
        o.mask *= hmi_magnetogram.data_mask[y:y + h, x:x + w]
        cos_map = hmi_magnetogram.get_cosine_map()
        o.cos_map = cos_map[y:y + h, x:x + w]
        hg_lont_map, hg_latd_map = \
            hmi_magnetogram.get_heliographic_coordinates()

        # feature characterization: arrays-------------------------------------
        data = np.ma.array(hmi_magnetogram.data[y:y + h, x:x + w] * o.mask,
                           mask=1 - o.mask)
        data *= (~cv2.inRange(data,
                              -params.STATIC_BACKGROUND_THRESHOLD,
                              params.STATIC_BACKGROUND_THRESHOLD)) & 1
        o.abs_data = np.ma.absolute(data)
        o.hg_longitude_map = np.ma.array(hg_lont_map[y:y + h, x:x + w],
                                         mask=1 - o.mask)
        o.hg_latitude_map = np.ma.array(hg_latd_map[y:y + h, x:x + w],
                                        mask=1 - o.mask)

        o.area_map = np.ma.array(
            (o.mask / o.cos_map) * params.AREA_PER_PIXEL, mask=1 - o.mask)
        o.phi_map = np.ma.array(data * 10 ** 4 * o.area_map,
                                mask=1 - o.mask)  # 1 G = 1 Mx/cm^2, 1 Gauss
        if delta_t_magnetogram is None:
            o.phi_delta = np.zeros(data.shape)
        else:
            # the shapes may not be euqal, so we take the maximum area of both
            delta_phi_map = delta_t_magnetogram.data[y:y + h,
                                                     x:x + w] * o.area_map
            o.phi_delta = (np.ma.absolute(o.phi_map) - np.ma.absolute(
                delta_phi_map)) / delta_t

        # feature characterization: properties---------------------------------
        o.max = np.ma.amax(data)
        o.min = np.ma.amin(data)
        o.sum = np.ma.sum(data)
        o.abs_sum = np.ma.sum(o.abs_data)
        hg_longitude = np.ma.sum(o.abs_data * o.hg_longitude_map) / o.abs_sum
        hg_latitude = np.ma.sum(o.abs_data * o.hg_latitude_map) / o.abs_sum
        o.position = {"longitude": hg_longitude, "latitude": hg_latitude}
        o.mean = np.ma.mean(data)
        o.variance = np.ma.var(data)
        o.skewness = scipy.stats.mstats.skew(data, axis=None)
        o.kurtosis = scipy.stats.mstats.kurtosis(data, axis=None)
        o.area = np.ma.sum(o.area_map)

        o.phi_pos = np.ma.sum(o.phi_map[o.phi_map > 0])
        if o.phi_pos is np.ma.masked:
            o.phi_pos = 0
        o.phi_neg = np.ma.sum(o.phi_map[o.phi_map < 0])
        if o.phi_neg is np.ma.masked:
            o.phi_neg = 0
        o.phi_abs = np.ma.sum(np.ma.absolute(o.phi_map))
        o.phi_imb = abs(o.phi_pos - abs(o.phi_neg)) / o.phi_abs
        o.phi_net_emrg = np.sum(o.phi_delta)

        # polarity separation lines--------------------------------------------
        r = params.PSL_DILATION_RADIUS
        kernel = img_operations.create_circle_mask(r, (r, r), (2 * r, 2 * r))
        positive = cv2.dilate(
            cv2.threshold(data, 1, 1, cv2.THRESH_BINARY)[1], kernel)
        negative = cv2.dilate(
            cv2.threshold(data, -1, 1, cv2.THRESH_BINARY_INV)[1], kernel)
        o.psl_mask = positive * negative

        # from http://opencvpython.blogspot.ch/2012/05/skeletonization-using-
        # opencv-python.html
        orig_mask = o.psl_mask.astype(np.uint8)
        size = np.size(o.psl_mask)
        o.psl_thin_mask = np.zeros(o.psl_mask.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(orig_mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(orig_mask, temp)
            o.psl_thin_mask = cv2.bitwise_or(o.psl_thin_mask, temp)
            orig_mask = eroded.copy()
            zeros = size - cv2.countNonZero(orig_mask)
            if zeros == size:
                done = True
        # end from opencvpython blogspot

        o.PSL_len = np.sum(o.psl_thin_mask)
        gradient_map = cv2.Sobel(data, -1, 1, 1)
        gradient_per_area = (
            gradient_map / (params.METERS_PER_PIXEL / 10 ** 6)
        )
        o.SG_len = np.sum(
            o.psl_thin_mask * (gradient_per_area > params.SG_THRESHOLD)
        )
        o.WL_sg_star = np.sum(o.psl_thin_mask * gradient_map)
        r_star_map = data * cv2.GaussianBlur(o.psl_mask, (
            params.GAUSSIAN_BLUR_KERNEL_SIZE * 2 + 1,
            params.GAUSSIAN_BLUR_KERNEL_SIZE * 2 + 1),
            16.8)
        o.R_star = np.sum(r_star_map)

        return o

    @classmethod
    def from_json(cls, _dict):
        o = cls()
        o.id = _dict["fc_id"]
        o.time = iso8601.parse_date(_dict["time_start"])
        o.position = {"latitude": _dict["lat_hg"],
                      "longitude": _dict["long_hg"]}
        o.index = _dict["data"]["index"]
        o.shape = {
            "x": int(_dict["data"]["pos_x"]),
            "y": int(_dict["data"]["pos_y"]),
            "width": int(_dict["data"]["width"]),
            "height": int(_dict["data"]["height"])
        }
        o.contour = np.array(json.loads(_dict["data"]["contour"]))
        o.mask = np.zeros((o.shape["height"], o.shape["width"]),
                          dtype=np.uint8)
        cv2.drawContours(o.mask,
                         [o.contour - (o.shape["x"], o.shape["y"])],
                         0,
                         1,
                         -1)
        o.max = _dict["data"]["max"]
        o.min = _dict["data"]["min"]
        o.sum = _dict["data"]["sum"]
        o.abs_sum = _dict["data"]["abs_sum"]
        o.mean = _dict["data"]["mean"]
        o.variance = _dict["data"]["variance"]
        o.skewness = _dict["data"]["skewness"]
        o.kurtosis = _dict["data"]["kurtosis"]
        o.area = _dict["data"]["area"]
        o.phi_pos = _dict["data"]["phi_pos"]
        o.phi_neg = _dict["data"]["phi_neg"]
        o.phi_abs = _dict["data"]["phi_abs"]
        o.phi_imb = _dict["data"]["phi_imb"]
        o.phi_net_emrg = _dict["data"]["phi_net_emrg"]
        o.PSL_len = _dict["data"]["PSL_len"]
        o.SG_len = _dict["data"]["SG_len"]
        o.R_star = _dict["data"]["R_star"]
        o.WL_sg_star = _dict["data"]["WL_sg_star"]

        return o

    def json(self):
        return {
            "time_start": self.time.isoformat() + "Z",
            "lat_hg": self.position["latitude"],
            "long_hg": self.position["longitude"],
            "nar": 0,
            "data": {
                "index": self.index,
                "contour": self.contour.tolist(),
                "pos_x": int(self.shape["x"]),
                "pos_y": int(self.shape["y"]),
                "width": int(self.shape["width"]),
                "height": int(self.shape["height"]),
                "max": float(self.max),
                "min": float(self.min),
                "sum": float(self.sum),
                "abs_sum": float(self.abs_sum),
                "mean": float(self.mean),
                "variance": float(self.variance),
                "skewness": float(self.skewness),
                "kurtosis": float(self.kurtosis),
                "area": float(self.area),
                "phi_pos": float(self.phi_pos),
                "phi_neg": float(self.phi_neg),
                "phi_abs": float(self.phi_abs),
                "phi_imb": float(self.phi_imb),
                "phi_net_emrg": float(self.phi_net_emrg),
                "PSL_len": int(self.PSL_len),
                "SG_len": int(self.SG_len),
                "R_star": float(self.R_star),
                "WL_sg_star": float(self.WL_sg_star),
                "class": self.classification()
            }
        }

    def get_shape(self):
        return (
            self.shape["x"],
            self.shape["y"],
            self.shape["width"],
            self.shape["height"]
        )

    def classification(self):
        if self.phi_imb > 0.9:
            polarity_balance = "Unipolar"
        else:
            polarity_balance = "Multipolar"
        if self.phi_abs > 1.0e21:
            size = "Large"
        else:
            size = "Small"
        if self.phi_net_emrg >= 0:
            growth = "Emerging"
        else:
            growth = "Decaying"
        return polarity_balance[0] + size[0] + growth[0]

    def __repr__(self):
        return str(self.id)
