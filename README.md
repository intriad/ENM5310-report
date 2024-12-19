# ENM5310-report

This is the code and some example data that was generated for the report. Here is the file structure breakdown:

**`/code/`**

The code directory includes all Python scripts and Jupyter notebooks that were used to run all calculations and generate figures.

"\

**`/code/adam_linear_regression.py`**

This script includes a linear regression class with the Adam optimizer to find optimal linear model parameters. It takes in a feature list file (any from /mxene_dataset/) and a data file (any from /mxene_dataset/). Writes loss and training/testing data to a .csv file. The directory where the .csv is saved must be adjusted per user, and all results are saved in /nolr_results/.


**`/code/neuralnet.py`**

Similarly takes a feature list file (from /mxene_dataset/) and data file (from /mxene_dataset/) and writes .csv results to /mlp_results/


**`/code/sisso.py`**

This is the Python scikit-learn wrapper for the SISSO++ package. SISSO++ MUST be compiled on the machine from http://gitlab.com/sissopp_developers/sissopp. This code also takes in a feature list file and a data file from /mxene_dataset/ and runs the SISSO regressor. The operator set must be defined in the sisso.py script. NOTE: the sisso.py script was run on the Chestnut computing cluster - the submission scripts exist in the /sisso_results/ directory. Also note that the SISSO model results are saved in the sisso.py working directory itself, so each final model is saved in its own subdirectory (named for adsorbate and feature set) in /sisso_results/


**`/code/*.ipynb`**

Any Jupyter notebook in the /code/ directory is a helper function that was used to conveniently run all scripts, submit all jobs, plot all figures, etc.


**`/mlp_results/`**

The .csv files with loss and training/testing data/predictions from the neural network.


**`/mxene_dataset/`**

Any .txt file is a feature set file. Any .csv file is the MXene adsorption energy or feature data. These data are what all the /code/ python scripts reference.


**`/nolr_results/`**

Adam-optimized linear regression .csv results.


**`/sisso_results/`**

All SISSO++ model results.

**`/sisso_results/allops/`**

The full operator set as referenced in the final report. We only include this set of results in the repository for simplicity to give an example of how the SISSO++ file structure works.
