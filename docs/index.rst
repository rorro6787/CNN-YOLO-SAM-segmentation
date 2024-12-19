.. algorithm_benchmark_toolkit documentation master file, created by
   sphinx-quickstart on Thu Nov 28 10:42:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CNN Methods
===========================

This is the documentation for the `cnn-methods` python library, which is a python project explores the application of advanced computer vision techniques for fruit classification and segmentation. A custom Convolutional Neural Network (CNN) was designed and implemented to classify different types of fruits, focusing on achieving high accuracy with an efficient architecture. The CNN was trained on a labeled dataset of fruit images, utilizing techniques such as data augmentation and optimization strategies to enhance performance and robustness.

Additionally, the project integrates a YOLO (You Only Look Once) model for real-time object detection and leverages the combination of YOLO and SAM (Segment Anything Model) for instance segmentation. This enables precise identification and segmentation of individual fruit instances, including segmentation with bounding boxes for more detailed analysis. The combination of custom and pre-trained models demonstrates versatility and effectiveness across multiple computer vision tasks.

Installation
============

Before installing, your system must satisfy these minimum requirements:

- **Python**: >=3.8
- **Linux / MacOS**: Windows is not supported because the triton package is not available for Windows and we use it for GPU acceleration. We recommend Google Colab to Windows users.


To install the project, you need to clone the repository and install the required dependencies. You will need to have Python 3.8 or higher installed on your system. Before installing the project, we recommend creating a virtual environment to avoid conflicts with other Python projects:

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

If you want to use the CNN visualization tools, you will need to install the `graphviz` package. You can install it using the following command:

.. code-block:: bash

   sudo apt-get install graphviz

.. warning:: 
   **Beta Version**  
   This project is currently in **beta version**. and still has work in progress. We recommend referring to the section  `How to Jupyter <configure/configuration.html>`_  for detailed instructions.

Once you have activated the virtual environment, you can install the project dependencies using the following command:

.. code-block:: bash

   pip install cnn-methods

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   configure/configuration
   API/api
