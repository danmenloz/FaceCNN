# FaceCNN
Face Detection using a Convolutional Neural Network in Pytorch

## Setup Information
To run this project follow, first clone the repository and follow the next steps:

1. To download the original images from the faceSrub dataset, run the following command:

python ./faceScrub download.py

2. Run the **main.py** script using for example the following command:

 python main.py --epochs 10 --lr 0.015 --l2 0.01 --momentum 0.9 --lr_decay_rate 0.1 --verbose 3 --suffix aug

3. Open the **main.py** script to see all the available options.

3. Open the **Project03.ipynb** notebook to see the results discussed in our report.

This project was built using the following anaconda environment:                
channels:
  - pytorch
  - defaults
dependencies:
  - appnope=0.1.2=py38hecd8cb5_1001
  - argon2-cffi=20.1.0=py38h9ed2024_1
  - async_generator=1.10=pyhd3eb1b0_0
  - attrs=20.3.0=pyhd3eb1b0_0
  - backcall=0.2.0=pyhd3eb1b0_0
  - blas=1.0=mkl
  - bleach=3.3.0=pyhd3eb1b0_0
  - ca-certificates=2021.4.13=hecd8cb5_1
  - certifi=2020.12.5=py38hecd8cb5_0
  - cffi=1.14.5=py38h2125817_0
  - cycler=0.10.0=py38_0
  - dbus=1.13.18=h18a8e69_0
  - decorator=5.0.6=pyhd3eb1b0_0
  - defusedxml=0.7.1=pyhd3eb1b0_0
  - entrypoints=0.3=py38_0
  - expat=2.3.0=h23ab428_2
  - freetype=2.10.4=ha233b18_0
  - gettext=0.21.0=h7535e17_0
  - glib=2.68.1=hdf23fa2_0
  - icu=58.2=h0a44026_3
  - importlib-metadata=3.10.0=py38hecd8cb5_0
  - importlib_metadata=3.10.0=hd3eb1b0_0
  - intel-openmp=2019.4=233
  - ipykernel=5.3.4=py38h5ca1d4c_0
  - ipython=7.22.0=py38h01d92e1_0
  - ipython_genutils=0.2.0=pyhd3eb1b0_1
  - ipywidgets=7.6.3=pyhd3eb1b0_1
  - jedi=0.17.0=py38_0
  - jinja2=2.11.3=pyhd3eb1b0_0
  - joblib=1.0.1=pyhd3eb1b0_0
  - jpeg=9b=he5867d9_2
  - jsonschema=3.2.0=py_2
  - jupyter=1.0.0=py38_7
  - jupyter_client=6.1.12=pyhd3eb1b0_0
  - jupyter_console=6.4.0=pyhd3eb1b0_0
  - jupyter_core=4.7.1=py38hecd8cb5_0
  - jupyterlab_pygments=0.1.2=py_0
  - jupyterlab_widgets=1.0.0=pyhd3eb1b0_1
  - kiwisolver=1.3.1=py38h23ab428_0
  - lcms2=2.12=hf1fd2bf_0
  - libcxx=10.0.0=1
  - libffi=3.3=hb1e8313_2
  - libgfortran=3.0.1=h93005f0_2
  - libiconv=1.16=h1de35cc_0
  - libpng=1.6.37=ha441bb4_0
  - libsodium=1.0.18=h1de35cc_0
  - libtiff=4.1.0=hcb84e12_1
  - libuv=1.40.0=haf1e3a3_0
  - libxml2=2.9.10=h7cdb67c_3
  - llvm-openmp=10.0.0=h28b9765_0
  - lz4-c=1.9.3=h23ab428_0
  - markupsafe=1.1.1=py38h1de35cc_1
  - matplotlib=3.3.4=py38hecd8cb5_0
  - matplotlib-base=3.3.4=py38h8b3ea08_0
  - mistune=0.8.4=py38h1de35cc_1001
  - mkl=2019.4=233
  - mkl-service=2.3.0=py38h9ed2024_0
  - mkl_fft=1.3.0=py38ha059aab_0
  - mkl_random=1.1.1=py38h959d312_0
  - nbclient=0.5.3=pyhd3eb1b0_0
  - nbconvert=6.0.7=py38_0
  - nbformat=5.1.3=pyhd3eb1b0_0
  - ncurses=6.2=h0a44026_1
  - nest-asyncio=1.5.1=pyhd3eb1b0_0
  - ninja=1.10.2=hf7b0b51_1
  - notebook=6.3.0=py38hecd8cb5_0
  - numpy=1.19.2=py38h456fd55_0
  - numpy-base=1.19.2=py38hcfb5961_0
  - olefile=0.46=py_0
  - openssl=1.1.1k=h9ed2024_0
  - packaging=20.9=pyhd3eb1b0_0
  - pandas=1.2.4=py38h23ab428_0
  - pandoc=2.12=hecd8cb5_0
  - pandocfilters=1.4.3=py38hecd8cb5_1
  - parso=0.8.2=pyhd3eb1b0_0
  - pcre=8.44=hb1e8313_0
  - pexpect=4.8.0=pyhd3eb1b0_3
  - pickleshare=0.7.5=pyhd3eb1b0_1003
  - pillow=8.2.0=py38h5270095_0
  - pip=21.0.1=py38hecd8cb5_0
  - prometheus_client=0.10.1=pyhd3eb1b0_0
  - prompt-toolkit=3.0.17=pyh06a4308_0
  - prompt_toolkit=3.0.17=hd3eb1b0_0
  - ptyprocess=0.7.0=pyhd3eb1b0_2
  - pycparser=2.20=py_2
  - pygments=2.8.1=pyhd3eb1b0_0
  - pyparsing=2.4.7=pyhd3eb1b0_0
  - pyqt=5.9.2=py38h655552a_2
  - pyrsistent=0.17.3=py38haf1e3a3_0
  - python=3.8.8=h88f2d9e_5
  - python-dateutil=2.8.1=pyhd3eb1b0_0
  - pytorch=1.7.0=py3.8_0
  - pytz=2021.1=pyhd3eb1b0_0
  - pyzmq=20.0.0=py38h23ab428_1
  - qt=5.9.7=h468cd18_1
  - qtconsole=5.0.3=pyhd3eb1b0_0
  - qtpy=1.9.0=py_0
  - readline=8.1=h9ed2024_0
  - scikit-learn=0.24.1=py38hb2f4e1b_0
  - scipy=1.6.2=py38h2515648_0
  - seaborn=0.11.1=pyhd3eb1b0_0
  - send2trash=1.5.0=pyhd3eb1b0_1
  - setuptools=52.0.0=py38hecd8cb5_0
  - sip=4.19.8=py38h0a44026_0
  - six=1.15.0=py38hecd8cb5_0
  - sqlite=3.35.4=hce871da_0
  - terminado=0.9.4=py38hecd8cb5_0
  - testpath=0.4.4=pyhd3eb1b0_0
  - threadpoolctl=2.1.0=pyh5ca1d4c_0
  - tk=8.6.10=hb0a8c7a_0
  - torchaudio=0.7.0=py38
  - torchvision=0.8.0=py38_cpu
  - tornado=6.1=py38h9ed2024_0
  - tqdm=4.59.0=pyhd3eb1b0_1
  - traitlets=5.0.5=pyhd3eb1b0_0
  - typing_extensions=3.7.4.3=pyha847dfd_0
  - wcwidth=0.2.5=py_0
  - webencodings=0.5.1=py38_1
  - wheel=0.36.2=pyhd3eb1b0_0
  - widgetsnbextension=3.5.1=py38_0
  - xz=5.2.5=h1de35cc_0
  - zeromq=4.3.4=h23ab428_0
  - zipp=3.4.1=pyhd3eb1b0_0
  - zlib=1.2.11=h1de35cc_3
  - zstd=1.4.9=h322a384_0
  - pip:
    - torchsummary==1.5.1