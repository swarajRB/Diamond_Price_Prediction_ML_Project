# Importing necessary modules from sklearn
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # For scaling and encoding
from sklearn.pipeline import Pipeline  # For creating ML pipelines
from sklearn.compose import ColumnTransformer  # For handling different types of columns

# Other standard imports
import sys, os
from dataclasses import dataclass  # For cleaner config classes
import numpy as np
import pandas as pd

# Importing custom exception and logger
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # Function to save objects using pickle


# Configuration class to define the path where the preprocessor object will be saved
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# Main class for Data Transformation
class DataTransformation:
    def __init__(self):
        # Instantiate configuration
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        """
        This function creates and returns a preprocessing object that
        handles missing values, encoding, and scaling for both numerical and categorical features.
        """
        try:
            logging.info('Data Transformation initiated')

            # Columns by type
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Defining the custom order for ordinal encoding
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Pipeline Initiated')

            # Pipeline for numerical columns
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
                ('scaler', StandardScaler())  # Standardize features
            ])

            # Pipeline for categorical columns
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with most frequent value
                ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())  # Scale encoded values
            ])

            # Combining numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function applies the transformation pipeline on training and testing data,
        and saves the preprocessor object to disk.
        """
        try:
            # Reading train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            # Get the preprocessor pipeline
            preprocessing_obj = self.get_data_transformation_object()

            # Define target and features
            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']  # Columns to drop from input features

            # Splitting train data
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Splitting test data
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Applying transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing object on training and testing datasets.")

            # Concatenate the transformed input features with their respective target values
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor pipeline to a .pkl file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved successfully.')

            # Return transformed arrays and file path to preprocessor
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.info("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)
