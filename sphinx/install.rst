.. _install:


.. ::

    # with overline, for parts
    * with overline, for chapters
    =, for sections
    -, for subsections
    ^, for subsubsections
    ", for paragraphs

Install
*********

Requirements
=============

* Python 2.7.13 (e.g., from Anaconda) with pip.
* `Git LFS`_

.. _Git LFS: https://git-lfs.github.com/


Installation
============

Clone the repository from github and install the library and its dependencies:

.. code::
  
  git clone git@github.com:NREL/OpenOA.git
  pip install -e ./OpenOA


Example Data
============

Git-LFS is used to store large data files. To obtain these files, run the following commands:

.. code::
  
  git lfs pull

