This engineering diploma project is designed for web-based image transformation.
Here are multiple image transformation methods implemented.

1. First of all, to startup the main application, the api/backend_application.py file should be started. It starts up the backend and also give a possibility to access the frontend via localhost:8000. At this point all of the features should be accessible.

2. To prepare this functionality, multiple steps were completed, these steps are contained in the training and testing files.
3. blurring/TestBlurring.ipynb contains the code to display a plot for different blurring techniques
4. compression/CompressionService.py contains compression logic for the backend
5. compression/TestCompression.ipynb shows a plot for compressing the image of a big size
6. The dataset directory should contain all of the datasets to be able to run training loops. Because of their size, they cannot be attached to the project and should be downloaded using following instructions:
-VOC2012 dataset should be downloaded from https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset and obtained directories VOC2012_test and VOC2012_train_val should be put separately to dataset directory
-DIV2K dataset should be downloaded from https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images and obtained directories DIV2K_train_HR and DIV2K_valid_HR should be put separately to dataset directory
the overall structure of dataset directory should look like this: main directory (dataset) -> 4 child directories (VOC2012_test, VOC2012_train_val, DIV2K_train_HR, DIV2K_valid_HR)
additional illustrative structure of dataset directory scheme can be find in dataset-structure.jpg
7. The front_end directory contains the HTML and JS files for the web application page
8. unet_impl directory contains file with .pth extension where the results of different training approaches are stored
9. unet_impl/backbone_impl.py contains the training loop for the U-Net backbone
10. unet_impl/BackgroundBlurModel.py is used as a service for backend application
11. unet_impl/PascalVOC_Dataset.py contains custom dataset implementation to load dataset for training
12. unet_impl/test_with_image.py calculates the output mask of a test image
13. unet_impl/Unet_training.ipynb trains the model, and plots the training loss and accuracy, those metrics are stored in unet_results.pkl in order to future reproduction
14. unet_impl/unet_validation.py runs the evaluation of the model to give evaluation accuracy
15. upscaling_model/dataset_loader.py is used for custom dataset implementation to use as input to FSRCNN model
16. upscaling_model/FSRCNN.py is a custom class to define the FSRCNN model
17. upscaling_model/fsrcnn_training.ipynb runs training of the model, displays the training loss, and finally tests upscaling functionality on the test image
18. upscaling_model/UpscalingModel.py is used as backend service for upscaling
