# Monolayer Cardiomyocyte Beating Frequency Analysis

This Python application analyzes the beating frequency of monolayer cardiomyocytes in video recordings. The application supports gunner farnebäck optical flow-based methods for contractility measurement and provides options for customizable input and output file handling.



## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
  - [Function A: Optical Flow Based Method](#function-a-optical-flow-based-method)
- [License](#license)

## Overview

This tool analyzes videos of monolayer cardiomyocyte contractions to assess beating frequency and other metrics of contractility. It uses video processing techniques and an optical flow-based approach to track and measure movement over time, allowing users to calculate the frequency and characteristics of cardiomyocyte contractions.

## Features
- **Video-based Analysis**: Supports both `.mp4` and `.avi` video formats.
- **Optical Flow Tracking**: Calculates beating frequency through motion tracking.
- **Customizable Output**: Options for saving processed videos to specified output paths.

## Installation

This application requires a Conda environment to manage dependencies. Use the provided `environment.yaml` file to create the environment. Ensure you have Conda installed.

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/monolayer-cardiomyocyte-frequency-analysis.git
    cd monolayer-cardiomyocyte-frequency-analysis
    ```

2. Create and activate the Conda environment:
    ```bash
    conda env create -f environment.yaml
    conda activate cardiomyocyte-analysis
    ```

## Usage

To run the application, use the following command structure:

```bash
python main.py base -i path/to/input_video.mp4 -ov path/to/output_video.mp4
