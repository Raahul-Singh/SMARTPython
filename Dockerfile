# setup ipython notebook server
FROM flarecast/python-base
MAINTAINER i4Ds

RUN apt-get update && apt-get install -y cmake \
libblas-dev \
liblapack-dev \
libopencv-core2.4 \
python-opencv \
gfortran \
libpng12-dev \
libjpeg62-turbo-dev \
libjasper-dev \
libopenexr-dev \
libtiff5-dev \
libwebp-dev \
libtbb-dev \
libeigen3-dev \
 && apt-get autoremove -y && apt-get clean autoclean && rm -rf /var/lib/{apt,dpkg,cache,log}/

ADD ./ /code/

WORKDIR /code

ENV PYTHONPATH /usr/lib/python2.7/dist-packages
RUN ln -s /usr/lib/python2.7/dist-packages/cv2.x86_64-linux-gnu.so /usr/lib/python2.7/dist-packages/cv2.so

LABEL eu.flarecast.type="algorithm"
LABEL eu.flarecast.name="smart"
LABEL eu.flarecast.algorithm.pipe="extraction"

CMD [ "python", "-u", "-B", "./main.py" ]
