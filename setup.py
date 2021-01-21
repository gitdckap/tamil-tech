import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "tensorflow>=2.4.0",
    "torch",
    "torchaudio",
    "torchvision",
    "Open-Tamil",
    "pyaudio",
    "librosa",
    "tensorflow_datasets",
    "tensorflow-addons",
    "warp-rnnt",
    "nlpaug",
    "visdom",
    "fastapi",
    "uvicorn",
    "Flask",
    "logmmse",
    "soundfile",
    "werkzeug",
    "TensorflowASR",
    "uWSGI"
]

setuptools.setup(
    name="tamil_tech",
    version="0.0.1",
    author="DCKAP",
    author_email="apps@dckap.com",
    description="Tamil Tech is an Open-Source Library for end-to-end Natural Language Processing & Speech Processing for Tamil language, developed by DCKAP aiming to provide a common platform that allows developers, students, researchers, and experts around the world to build exciting AI based language and speech applications for Tamil language.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gitdckap/tamil-tech",
    packages=setuptools.find_packages(include=["tamil_tech*"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.6',
)