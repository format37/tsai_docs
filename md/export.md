## On this page

  * get_script_path
  * nb_name_to_py
  * get_nb_path
  * get_colab_nb_name
  * get_nb_name
  * nb2py



  * __Report an issue



# nb2py

> nb2py will allow you to convert the notebook (.ipynb) where the function is executed to a python script.

The conversion applies these rules:

  * The notebook will be automatically saved when the function is executed.
  * Only code cells will be converted (not markdown cells).
  * A header will be added to indicate the script has been automatically generated. It also indicates where the original ipynb is.
  * Cells with a #hide flag won’t be converted. Flag variants like # hide, #Hide, #HIDE, … are also acceptable.
  * Empty cells and unnecessary empty lines within cells will be removed.
  * By default the script will be created with the same name and in the same folder of the original notebook. But you can pass a dir folder and a different name if you wish.
  * If a script with the same name already exists, it will be overwriten.



This code is required to identify flags in the notebook. We are looking for #hide flags.

This code automatically gets the name of the notebook. It’s been tested to work on Jupyter notebooks, Jupyter Lab and Google Colab.

* * *

source

### get_script_path

> 
>      get_script_path (nb_name=None)

* * *

source

### nb_name_to_py

> 
>      nb_name_to_py (nb_name)

* * *

source

### get_nb_path

> 
>      get_nb_path ()

_Returns the absolute path of the notebook, or raises a FileNotFoundError exception if it cannot be determined._

* * *

source

### get_colab_nb_name

> 
>      get_colab_nb_name ()

* * *

source

### get_nb_name

> 
>      get_nb_name (d=None)

_Returns the short name of the notebook w/o the .ipynb extension, or raises a FileNotFoundError exception if it cannot be determined._

This code is used when trying to save a file to google drive. We first need to mount the drive.

* * *

source

### nb2py

> 
>      nb2py (nb:str<absoluteorrelativefullpathtothenotebookyouwanttoconverttoap
>             ythonscript>=None, folder:str<absoluteorrelativepathtofolderofthes
>             criptyouwillcreate.Defaultstocurrentnb'sdirectory>=None, name:str<
>             nameofthescriptyouwanttocreate.Defaultstocurrentnbname.ipynbby.py>
>             =None, save:<savesthenbbeforeconvertingittoascript>=True,
>             run:<importandrunthescript>=False,
>             verbose:<controlsverbosity>=True)

_Converts a notebook to a python script in a predefined folder._
    
    
    if not is_colab():
        nb = None
        folder = None
        name = None
        pyname = nb2py(nb=nb, folder=folder, name=name)
        if pyname is not None: 
            assert os.path.isfile(pyname)
            os.remove(pyname)
            assert not os.path.isfile(pyname)
    
        nb = '001_export.ipynb'
        folder = None
        name = None
        pyname = nb2py(nb=nb, folder=folder, name=name)
        if pyname is not None: 
            assert os.path.isfile(pyname)
            os.remove(pyname)
            assert not os.path.isfile(pyname)
    
        nb = '../nbs/001_export'
        folder = None
        name = None
        pyname = nb2py(nb=nb, folder=folder, name=name)
        if pyname is not None: 
            assert os.path.isfile(pyname)
            os.remove(pyname)
            assert not os.path.isfile(pyname)
    
        nb = None
        folder = '../test_export/'
        name = None
        pyname = nb2py(nb=nb, folder=folder, name=name)
        if pyname is not None: 
            assert os.path.isfile(pyname)
            shutil.rmtree(folder)
            assert not os.path.isfile(pyname)__
    
    
    nb2py couldn't get the nb name. Pass it as an nb argument and rerun nb2py.
    001_export.ipynb converted to /Users/nacho/notebooks/tsai/nbs/001_export.py
    001_export.ipynb converted to /Users/nacho/notebooks/tsai/nbs/../nbs/001_export.py
    nb2py couldn't get the nb name. Pass it as an nb argument and rerun nb2py.

  * __Report an issue


