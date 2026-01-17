# PYNQZ2-CNN

This project consists in the implementation of a Convolutional Neural Network (CNN) on a PYNQ-Z2 FPGA, using HLS4ML, Keras, and TensorFlow. The goal is to accelerate the inference of the CNN on the FPGA.

## Dataset

The dataset used in this project is the RML2016.10a, created by [CREATORS OF THE DATASET]. The data has been prepared by viewing and treating it to be suitable for the CNN. The dataset consists of radio signals with different modulations.

## Installation

This project uses Poetry for dependency management.

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Install the dependencies using Poetry:
    ```bash
    poetry install
    ```

## Usage

To run the project, you can use the following commands:

*   To plot the IQ constellation of the signals:
    ```bash
    poetry run python plot_iq.py
    ```
*   To plot the waveforms of the signals:
    ```bash
    poetry run python plot_waveforms.py
    ```

## Authors

*   [Your Name]
*   [Another Author]

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
