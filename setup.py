from setuptools import setup, find_packages

setup(
    name="BoT_1D_Model",                  
    version="0.1.0",
    description="1D reactive‐transport & flow solver for PWRI",
    package_dir={"": "BoT_1D_Model"},     
    packages=find_packages(where="BoT_1D_Model"),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    entry_points={                        
        "console_scripts": [
            "bot1d=main:main",           
        ],
    },
)
