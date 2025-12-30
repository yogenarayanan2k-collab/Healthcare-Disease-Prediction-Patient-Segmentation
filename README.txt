HEALTHCARE DISEASE PREDICTION – MACHINE LEARNING PROJECT
========================================================
PROJECT TITLE:

Healthcare Disease Prediction using Supervised and Unsupervised Machine Learning
================================================================================

PROJECT DESCRIPTION:

      This project focuses on predicting cardiovascular disease using patient
      health data.
      It applies Supervised Learning models for prediction and Unsupervised Learning for
      patient segmentation.

The project includes:

     * Data preprocessing
     * Feature engineering
     * Data visualization
     * Supervised ML models (Random Forest, SVM)
     * Unsupervised ML model (K-Means clustering)
     * Model evaluation and visualization

Dataset Used


     * Dataset Name: Cardio Vascular Disease Dataset
     * File: cardio_train.csv
     * Records: 70,000
     * Features: 13 input features + 1 target column
     * Target Variable: cardio
          0 → No heart disease
          1 → Heart disease present

Technologies & Libraries

      * Programming Language: Python 3
      * Libraries Used:
            pandas
            numpy
            matplotlib
            seaborn
            scikit-learn

Machine Learning Models Used

Supervised Learning:
      1. Random Forest Classifier
             Hyperparameter tuning using GridSearchCV

      2. Support Vector Machine (SVM)
             Kernel and regularization optimization

Unsupervised Learning:

      1. K-Means Clustering
             Patient segmentation
             PCA used for 2D visualization

Project Workflow
      1. Load and explore dataset
      2. Perform feature engineering
           Convert age from days to years
      3. Visualize target distribution
      4. Correlation heatmap
      5. Outlier visualization
      6. Data scaling using StandardScaler
      7. Train-test split
      8. Train supervised ML models
      9. Evaluate models using:
            Accuracy
            Confusion Matrix
            Classification Report
      10. Perform clustering using K-Means
      11. Visualize clusters using PCA

To Run the Project
Install required libraries
place given dataset
Run the program

Output:
    Model accuracy scores
    Confusion matrices
    Classification reports
    Heatmaps and plots
    PCA-based cluster visualization

   "Due to the large dataset size and hyperparameter tuning using GridSearchCV,
    the model training process takes additional execution time.
    This ensures better model optimization and improved prediction performance".

    The system successfully predicts cardiovascular disease using supervised machine learning models.
    Hyperparameter tuning improves accuracy,and confusion matrices are used for evaluation.
    K-Means clustering identifies patient groups,and PCA visualization provides insights into patient segmentation.

Conclusion:
     This project demonstrates how machine learning can be applied in healthcare to:
           Predict cardiovascular disease
           Analyze important health indicators
           Segment patients using unsupervised learning











