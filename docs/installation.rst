Installation
============

This package is still in its development phase, but you can compile the package from source

.. code-block:: bash

  git clone https://github.com/factoryofthesun/IVaps
  cd IVaps
  pip install .

To install in development mode

.. code-block:: bash

  git clone https://github.com/factoryofthesun/IVaps
  cd IVaps
  pip install -e ./

The installation will automatically detect whether there is a compatible GPU device on the system and install either onnxruntime or onnxruntime-gpu. Please note that the default onnxruntime GPU build requires CUDA runtime libraries being installed on the system. Please see the `onnxruntime repository <https://github.com/microsoft/onnxruntime>`_ for more details regarding the GPU build.
