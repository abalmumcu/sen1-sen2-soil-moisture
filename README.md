# Estimating Soil Moisture from Satellite Imagery

This project aims to estimate soil moisture from Sentinel 1 and Sentinel 2 imageries using machine learning techniques. By leveraging satellite data, we can obtain valuable insights into soil moisture levels, which are crucial for agricultural applications such as irrigation planning and crop monitoring.

## Dataset

The primary dataset used in this project is the Real-Time In-Situ Soil Monitoring for Agriculture (RISMA) dataset. This dataset provides ground-truth soil moisture measurements collected from various agricultural stations. Additionally, we also utilize an image dataset extracted for 13 stations in Manitoba from Google Earth Engine. These images serve as input data for training and testing our models.

## Directories

- `GAN`: Contains the code and resources related to the implementation of Generative Adversarial Networks (GANs) for data augmentation.
- `RISMA`: Includes the code and resources for data preprocessing, analysis, and visualization related to the RISMA dataset.
- `image_extraction`: Contains the code and resources for extracting images from Google Earth Engine for the Manitoba stations.
- `regression`: Includes the code and resources for the regression models used to estimate soil moisture.

## Usage

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/abalmumcu/sen1-sen2-soil-moisture.git`
2. Navigate to the specific directory you are interested in (e.g., `GAN`, `RISMA`, `image_extraction`, `regression`).
3. Follow the instructions provided in each directory's README file to set up and run the corresponding code.

## License

This project is licensed under the [MIT License](LICENSE).

