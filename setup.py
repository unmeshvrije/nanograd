import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nanograd",
    version="0.1.0",
    author="Unmesh Joshi",
    author_email="unmeshjoshiofficial@gmail.com",
    description="An educational fork of karpathy/micrograd: a tiny scalar-valued autograd engine with a small PyTorch-like neural network library on top, plus course lessons.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karpathy/micrograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
