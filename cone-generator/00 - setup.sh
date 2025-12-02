# created for linux setup
# https://learn.microsoft.com/en-us/windows/wsl/install
# ubuntu linux


# installation of dependencies
##################################

# freecad
sudo snap install freecad
export PATH="/snap/bin:$PATH"
export PATH="/snap/freecad/current/usr/lib:$PATH"



echo 'checking to ensure you have python installed'
echo python3 --version

# python stuff
python3-pip install numpy
python3-pip install matplotlib
python3-pip install scipy
python3-pip install trimesh
python3-pip install pyglet # might need older version? idk

