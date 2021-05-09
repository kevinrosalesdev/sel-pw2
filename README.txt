Structure & Contents of the ZIP File [RF & DF]
======================================================
- Documentation/
    * report.pdf: Report of the project containing the requested information such as
                  pseudo-code, evaluation of results, instructions on how to execute the code...

- Data/
    * slides.csv: Toy dataset used in the CART course slides.
    * lenses.csv: Toy dataset used in the RB-Classifiers course slides.
    * ecoli.csv: Small numerical dataset.
    * car.csv: Medium categorical dataset.
    * rice.csv: Large numerical dataset.
    * stroke.csv: Large mixed dataset.

- Out/
    * [DF/RF]-[dataset].csv: Accuracy results of the different datasets for different combination
                             of hyperparameters for both RF and DF, including the ordered list of
                             features and other metrics.

- Source/
    - datapreprocessor/
        * preprocessor.py: Loads and preprocesses the data from the datasets.
    - forests/
        * CART.py: implementation of CART.
        * DecisionForest.py: Implementation of Decision Forest.
        * RandomForest.py: Implementation of Random Forest.
    - utils/
        * metrics.py: Measures the quality of the results.

* main.py: Executable file. Plots the results of applying both RF and DF to each dataset. NT and F
           can be easily changed as parameters. The lines that call the function that generates
           the tables from 'Out/' folder are commented.

* README.txt: this file.