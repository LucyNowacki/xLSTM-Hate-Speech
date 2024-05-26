# xLSTM-Hate-Speech

This repository contains materials for a project focused on detecting hate speech using the xLSTM model.

## Repository Contents

- **DataSet**: Directory containing the dataset used for the project.
- **Data_Summary.ipynb**: Jupyter Notebook summarizing and analyzing the dataset.
- **Hate Speech Project.pdf**: A PDF document outlining the project's objectives, methodology, and results.
- **twitter-hate-speech-metadata.json**: Metadata for the Twitter hate speech dataset.

## Technology

- **PyTorch GPU**
- **Azure**

## Introduction

In this project, we aim to tackle the problem of hate speech detection using various machine learning approaches. Specifically, we will implement and compare two models:

- **NanoGPT**
- **xLSTM (Extended Long Short-Term Memory)**

We will train these models and compare their performance against state-of-the-art pre-trained models such as GPT, BERT, and ILAMA.

## Objective

The primary objective is to develop lightweight models that require significantly less memory and computational power, making them suitable for deployment on devices with limited resources like smartphones and tablets.

## Business Understanding

### Background

Hate speech on social media platforms and other online forums is a growing concern. Detecting and mitigating such harmful content is crucial to maintaining a safe and inclusive digital environment. Traditional large language models (LLMs) like GPT, BERT, and ILAMA are highly effective but require significant computational resources, making them impractical for deployment on resource-constrained devices.

### Goal

Our goal is to create efficient and lightweight models for hate speech detection that can be deployed on low-resource devices. This will enable real-time detection and filtering of harmful content without the need for powerful computational infrastructure.

In addition to efficiency, another aim is to build models with bespoke architectures that are relatively simple. These models should be easy to adjust to different platforms and scalable through parallelization. This flexibility will ensure that the models can be adapted for various deployment environments, enhancing their utility and longevity.

For future development beyond this project, the model should also be designed to accommodate not only language data but also visual data enriched by physical reality. This extension will involve integrating neural operators to handle continuous data and complex dependencies across different data modalities. By doing so, we aim to create a comprehensive solution capable of processing and understanding multi-modal data inputs, paving the way for more sophisticated applications.

## Benefits

- **Accessibility**: Enable hate speech detection on a wider range of devices, including smartphones and tablets.
- **Cost-Efficiency**: Reduce the need for expensive computational resources, making the technology more accessible to smaller organizations and developers.
- **Scalability**: Facilitate the deployment of models in resource-constrained environments, allowing for wider adoption and impact.
- **Real-Time Processing**: Allow for the real-time detection and mitigation of hate speech, enhancing the user experience and safety on digital platforms.
- **Flexibility**: The bespoke architecture ensures the model can be easily adjusted to different platforms, improving adaptability and deployment efficiency.
- **Future-Proofing**: The capability to integrate neural operators for handling both language and visual data ensures the model remains relevant and expandable for future applications involving multi-modal data inputs.

## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/LucyNowacki/xLSTM-Hate-Speech.git
    cd xLSTM-Hate-Speech
    ```

2. **Explore the dataset**:
    Navigate to the `DataSet` directory to familiarize yourself with the data.

3. **Run the data summary notebook**:
    Open the `Data_Summary.ipynb` file in Jupyter Notebook or JupyterLab to review the data analysis.

4. **Read the project documentation**:
    Refer to the `Hate Speech Project.pdf` for detailed information on the project's objectives, methodology, and results.

## Contributing

Contributions to this project are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.




