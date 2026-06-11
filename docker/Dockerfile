FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set LC_ALL and unset LANGUAGE
ENV LC_ALL=C
ENV LANGUAGE=

# Update, install dependencies and cleanup
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    git \
    build-essential \
    cmake \
    clang-tidy \
    libeigen3-dev \
    libproj-dev \
    libgeotiff-dev \
    libgtest-dev \
    libboost-all-dev \
    libgl1-mesa-glx \
    libegl1 libgl1 libgomp1 \
    python3-pip \
    python3-venv \
    vim \
    curl && \
    rm -rf /var/lib/apt/lists/*
    
# Create a directory for building packages
RUN mkdir -p /build && \
    cd /build

# Clone, build and clean up LASzip (3.x supports LAS 1.4 via laszip_api.h C API)
RUN git clone https://github.com/LASzip/LASzip.git && \
    cd LASzip && \
    git checkout tags/3.4.4 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && rm -rf LASzip

# Clone, build and clean up Qhull
RUN git clone https://github.com/qhull/qhull.git && \
    cd qhull && \
    git checkout tags/v7.3.2 && \
    cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=true && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && rm -rf qhull

# Clone, build and clean up libnabo
RUN git clone https://github.com/ethz-asl/libnabo.git && \
    cd libnabo && \
    git checkout tags/1.0.7 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && rm -rf libnabo

# Clone, build and install raycloudtools
RUN git clone https://github.com/csiro-robotics/raycloudtools.git && \
    cd raycloudtools && \
    mkdir build && \
    cd build && \
    cmake .. \
        -DGeoTIFF_INCLUDE_DIR=/usr/include/geotiff \
        -DGeoTIFF_LIBRARY=/usr/lib/x86_64-linux-gnu/libgeotiff.so \
        -DPROJ_INCLUDE_DIR=/usr/include/proj \
        -DPROJ_LIBRARY=/usr/lib/x86_64-linux-gnu/libproj.so \
        -DWITH_QHULL=ON \
        -DWITH_LAS=ON \
        -DDOUBLE_RAYS=ON \
        -DWITH_TIFF=ON \
        -DRAYCLOUD_BUILD_TESTS=ON \
        -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf raycloudtools

# Clone and build TreeTools
RUN git clone https://github.com/csiro-robotics/treetools.git && \
    cd treetools && \
    mkdir -p build && \
    cd build && \
    cmake .. \
        -DPROJ_INCLUDE_DIR=/usr/include/proj \
        -DPROJ_LIBRARY=/usr/lib/x86_64-linux-gnu/libproj.so \
        -DWITH_TIFF=ON \
        -DTREE_BUILD_TESTS=ON \
        -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf treetools

# Update ldconfig
RUN ldconfig /usr/local/lib

# Clean up build directory
RUN rm -rf /build

# Miniforge
ENV CONDA_DIR=/opt/conda
RUN curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Create dedicated runtime env with supported Python
RUN mamba create -n appenv -c conda-forge -y \
    python=3.11 \
    gdal \
    pdal \
    untwine \
    pip && \
    mamba clean --all -y

# Install Python packages into that env
RUN /opt/conda/envs/appenv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/conda/envs/appenv/bin/pip install --no-cache-dir \
        laspy \
        open3d \
        cloth-simulation-filter \
        geopandas \
        lazrs \
        pydantic \
        pydantic-settings \
        fiona \
        matplotlib \
        pyproj 

# Make the env default
ENV PATH=/opt/conda/envs/appenv/bin:/opt/conda/bin:$PATH

ENV PROJ_LIB=/opt/conda/envs/appenv/share/proj
ENV PROJ_DATA=/opt/conda/envs/appenv/share/proj

# Set the working directory
WORKDIR /workspace

COPY 3dtrees_Smart_Tile 3dtrees_Smart_Tile
COPY py-rct py-rct
COPY src /src

CMD ["/bin/bash"]