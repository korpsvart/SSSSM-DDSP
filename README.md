# Running training on Google Colab


For general informations look at the original repository. This branch contains code that can be used to train the model on Google Colab, plus some other useful scripts.

The problem of Colab is that you must load the data from Google Drive but if too many files are in the same folder it will time-out.
You can use `divide_datasets.py` to divide a large dataset into many sub-directories. Do this for both the in domain and the out of domain dataset (Magenta NSynth Dataset).
Then you must update the JSON of the out of domain dataset to account for the subdirectory information. You can do this with the `update_nsynth_json.py` script.


The Jupyter Notebook `SSSSM_DDSP_training.ipynb` shows an example of how we can then run the training on Colab, once the process above is completed.


## Export model via torchscript

If you want to export a model that will directly accept audio input and output a set of static parameters via Torchscript check out the branch `pytorch_scripting` branch and run the `torchscript_tracing.py`.
Details on the command to run are inside the file.