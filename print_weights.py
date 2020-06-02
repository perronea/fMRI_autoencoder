#!/usr/bin/env python

from __future__ import print_function

import h5py

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")

        print("  f.attrs.items(): ")
        for key, value in f.attrs.items():           
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            print("  Terminate # len(f.items())==0: ")
            return 

        print("  layer, g in f.items():")
        for layer, g in f.items():            
            print("  {}".format(layer))
            print("    g.attrs.items(): Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                print("    Dataset: param.keys():")
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()
print_structure('gordon_pconn_decoder_weights.h5')


