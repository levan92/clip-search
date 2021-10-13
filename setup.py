from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clip-search",
    version="0.1",
    author="levan92",
    author_email="lingevan0208@gmail.com",
    description="CLIP for forensic search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/levan92/clip-search",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(
        exclude=(
            "test",
            "examples",
        )
    ),
    # install_requires=[
    #     "numpy",
    #     "scipy",
    # ],
)
