This code includes code necessary to reproduce Fig. 1, 2, 3 and 5 of the accompanying submission.
The notebook ``kernels with guarantees examples.ipynb'' will reproduce figures 1, 5.
The notebook ``alignment and spectrum kernels.ipynb'' with reproduce figures 2, 3.

python version >= 3.6 is required to run these experiments as well as packages which can be install by running
pip install -r requirements.txt
This installs several packages including pyro and strkernel. To run the alignment kernel, an installation of pyro configured to one's GPU is required. As well, to use the kmer spectrum kernel we use the strkernel package at https://github.com/jakob-he/string-kernel under the MIT licence. The notebook alignment and spectrum kernels.ipynb must be run with a GPU to allow effcient evalutation of the alignment kernel. Make sure your pytorch installation is configured to your version of CUDA. This can be done at https://github.com/pytorch/pytorch#from-source.