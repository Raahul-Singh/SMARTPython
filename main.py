"""This is the main part of the S.M.A.R.T. algorithm. It will load the
necessary magnetograms from the server (set in config.py) and do the
extraction. The extracted features will be written back into the database.

"""
from __future__ import print_function


import urllib2
import requests
import json
import StringIO
from io import BytesIO
from flarecast.utils.property_db_client import PropertyDBClient

from hmi_magnetogram import HMIMagnetogram
from smart_feature import SMARTFeature
from native_rotation import native_rotation
try:
    import params1 as params
except:
    import params


print("S.M.A.R.T. info: libdc1394 errors are okay, they can be ignored\n")


def _url2magnetogram(url):
    img_stream = requests.get(url)
    img = BytesIO(img_stream.content)
    try:
        mag = HMIMagnetogram(img)
    except Exception as e:
        print("Error happend while downloading: %s" % e)
        mag = None
    return mag


def extract(start_, end_, hmiservice, propertyservice_url):
    client = PropertyDBClient(propertyservice_url)
    client.insert_provenances(["smart-python"])

    query_params = {
        "start": start_.isoformat() + "Z",
        "end": end_.isoformat() + "Z"
    }
    metadata = requests.get(hmiservice, params=query_params).json()

    if len(metadata) == 0:
        print('Warning: No images found in given time range (%s - %s)' % (
            start_, end_))
        return

    mag_t1 = _url2magnetogram(metadata[0]["url"])
    last_i = None

    for num, i in enumerate(metadata[1:]):
        print("processing magnetogram %d of %d (%s)" % (
            num + 1, len(metadata) - 1, i["date_obs"]))

        try:
            if mag_t1 is None:
                mag_t1 = _url2magnetogram(last_i["url"])
            mag_t0 = mag_t1
            mag_t1 = _url2magnetogram(i["url"])

            # differential rotation
            delta_time = (mag_t1.time - mag_t0.time).total_seconds()
            mag_t0.data = native_rotation.rotate(
                mag_t0.data,
                int(mag_t0.disk_center[0]),
                int(mag_t0.disk_center[1]),
                int(mag_t0.disk_radius),
                delta_time
            )

            # feature extraction
            contours = mag_t1.get_contours(mag_t0)

            features = []
            for j, contour in enumerate(contours):
                feature = SMARTFeature.from_hmi(mag_t1, j, contour,
                                                delta_time, mag_t0).json()
                features.append(feature)

            answer = client.insert_regions("smart-python", features)

            if "message" in answer:
                print("error while inserting property groups: %s" %
                      answer["message"])
                return

        except Exception as e:
            print("Error happend: %s" % e)
            mag_t1 = None

        last_i = i


if __name__ == '__main__':
    hmiservice = "http://hmi:8001/HMI/720"
    propertyservice_url = "http://property:8002"

    print("\n\nstart extraction")
    extract(params.START, params.END, hmiservice, propertyservice_url)
