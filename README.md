#e-dermatologist
An application based on Machine Learning to detect and classify skin disease for the underprivileged.
This model is a CNN (Convolutional Neaural Network) based model which consists of 7 Block stages of identifying the disease from an image.
The two datasets are used in this model DermNet (Kaggle) and ISIC 2019 (IEEE)
These two datasets are combined, cleaned, annotated, and balanced in order for the model to accurately identify the skin disease.
This model used multiple layers such as Dense, Dropout, Conv2D, Batch Normalization, MaxPooling2D, and Activation
The dataset currently contains five common skin diseases in the region.
Namely: 
    'Eczema':0,
    'Melanoma':1,
    'Acne':2,
    'Basal Cell Carcinoma':3,
    'Benign Keratosis':4
The model uses multiple tools: NumPy, TensorFlow, MatPlotLib, & SkLearn.'
We have used VGG, InceptionV3 but they do not perform better then Custom CNN model.
This model was developed through GoogleColab to obtain Virtual GPU processing in order to test the model more efficiently.
This model has acheived 90.24% accuracy till now.
