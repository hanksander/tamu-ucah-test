# setup
## install conda
 - go to https://www.anaconda.com/download
 - download and run the installer at https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
 - ensure that you **install for all users (systemwide)**
 - add conda to envronment variables
   - find where anaconda is located on your computer. For me, it was ```C:\ProgramData\anaconda3\Scripts```. This can be done by opening the anaconda terminal and typing where conda.
   - add to environment variables. Tutorial -> https://www.geeksforgeeks.org/python/how-to-setup-anaconda-path-to-environment-variable/
## set up a conda environment insid the working directory (stolen from the openvsp installation guide linked later)
1. Open up the Anaconda Prompt

2. Navigate to ```<Your OpenVSP Installation Path>/OpenVSP-<Version>-<OS>/python```

3. Run ```conda env create -f .\environment.yml```. This will create a Conda environment as specified in the environment.yml file.

4. Run conda activate vsppytools. This will activate the environment you just created. You should see (vsppytools) appear at the start of your terminal prompt.

5. Run ```pip install -r requirements-dev.txt```. This will set up all the python packages included with OpenVSP. As pip installs, you will see some depeciation warnings. At time of writing, these could be safely ignored and everything worked as expected.

6. Set the python interpreter in VS or Pycharm to be the conda envrionment


note: 
pip itself may not work in your conda terminal. If it is broken, you can work around it by using ```pyton -m pip``` ...

## activate conda in the terminal
 Open up a terminal 
 type "conda init" in the terminal
 command palet (>)
 createnew terminal

# use of the code
How to **Tailor the code** to the parameters you are interested in modifying.
  1) Open up the OpenVSP GUI and the file "Python all the way down.py"
  2) Navigate to the parameters you are interested in changing in OpenVSP. Click on the label next to the slider and a popup window should appear.
  3) Fill in the parameters in the file "Python all the way down.py" and run it
  4) Paste the output in the relevant places in the class Ogive (initialization goes in __init__, and the getter and setter functions go in the main body of the class
Class attributes and methods. Explanations are written as docstrings in each of the functions.
 - update_airframe_mesh()
 - add_mass()
 - graph_vehicle()
 - check_interferance()
 - save_vsp()
 - get_cfd_mesh()
 - get_cg()

# relevant resources

the python api documentation
https://openvsp.org/pyapi_docs/latest/openvsp.html

the installation guide
https://gist.github.com/HB-Stratos/afede7c04ddc5bf4f3f0f9f6dd8f5e4d

# notable file paths    

the file path for the Python files
 r"...\OpenVSP-3.46.0-win64\python"

 the file path for the vsp files:
 "..."



# to do:
 - component interferance detection
 - export CFD-quality mesh from python
 - import cbAero output data and weight the skin appropriately


# How to update the library
 - commit
 - push
 - update submodules for programs that depend on this code (ie, for tamu-ucah, go to tamu-ucah, open up a cmd prompt and enter "git submodule update --remote --merge")



 




