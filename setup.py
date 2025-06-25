from setuptools import setup

setup(
    install_requires=[
        "pandas==2.0.3",
        "streamlit==1.32.2"
    ],
    options={
        "bdist_wheel": {"universal": True}
    }
)
