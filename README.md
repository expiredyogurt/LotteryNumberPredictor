
# Lottery Predictor

This project is a machine learning model designed to predict the most likely set of lottery numbers for the next drawing based on previous winning numbers.

Modified the code to fit the template of sports toto's result template. 

Issue is all the generated numbers are between 30-55 without touching the bulk of the code from the main author

## Getting Started

To use this model, you will need to have Python installed on your computer, as well as the following libraries:

-   pandas
-   scikit-learn

To install the libraries, run the following command:

Copy code

`pip install pandas scikit-learn` 

## Usage

1.  Download the previous winning lottery numbers from sports toto in csv format and this code should run without issue
2.  Run the `Predictor.py` file, which will train a Random Forest Regression model on the previous winning numbers and generate a set of predicted numbers.
3.  The program will output the most likely set of numbers for the next drawing.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this project as long as you give attribution to the original author.
