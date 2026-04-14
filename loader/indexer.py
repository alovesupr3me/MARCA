import cooler
from scipy.sparse import coo_matrix
import math
import numpy as np
import pandas as pd
import math


class Indexer:
    """
    The class that indexes the contact map
    """

    def __init__(self, **kwargs):
        self.check_configs(**kwargs)
        self.idx_func = get_forward(self.args["filename"])

    def check_configs(self, **kwargs):
        """
        check all the settings in the config
        """
        if "filename" not in kwargs:
            raise ValueError("Missing required parameter: filename")
        if "resolution" not in kwargs:
            raise ValueError("Missing required parameter: resolution")        
                
        if "fetch_region" not in kwargs:
            raise ValueError("看看要说什么")    
            
        if "if_balanced" not in kwargs:
            print("Did not set if_balanced, we will not apply ICE normalization")
        self.if_balanced = kwargs.get("if_balanced", False)   
        
        if "chunk_size" not in kwargs:
            raise ValueError("Missing required parameter: chunk_size")        
        
        if "overlap_size" not in kwargs:
            raise ValueError("Missing required parameter: overlap_size")        

        # if "normalizer" not in kwargs:
        #     raise ValueError("Missing required parameter: normalizer")

        self.args = kwargs  

            

    def index(self):
        
        return self.idx_func(**self.args)

    
def get_forward(f):
    if f.endswith(".mcool"):
        func = idx_mcooler
    return func 


def idx_mcooler(**kwargs):
    fname = kwargs["filename"]
    resolution = int(kwargs["resolution"])
    region =  kwargs["fetch_region"]
    chunk_size = kwargs["chunk_size"]
    overlap_size = kwargs["overlap_size"]

    uri = '%s::/resolutions/%s' % (fname, resolution)
    clr = cooler.Cooler(uri)


    interacts_id = []
    for _, row in region.iterrows():
        
        chrom1, start1, end1, chrom2, start2, end2, name = row[
            ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "name"]
        ]


        chrom_size1, chrom_size2 = clr.chromsizes[chrom1], clr.chromsizes[chrom2]
        end1, end2 = min(end1, chrom_size1), min(end2, chrom_size2)


        start1 = start1 // resolution * resolution
        end1 = math.ceil(end1 / resolution) * resolution
        start2 = start2 // resolution * resolution
        end2 = math.ceil(end2 / resolution) * resolution


        chunk_starts1 = np.arange(start1, end1, (chunk_size - overlap_size) * resolution)  # step = (chunk_size - overlap_size) * resolution
        chunk_ends1 = np.minimum(chunk_starts1 + chunk_size * resolution, end1)  # end = start + chunk_size * resolution


        chunks_id1 = list(zip(
            np.full_like(chunk_starts1, chrom1, dtype=object),
            chunk_starts1,
            chunk_ends1,
            np.full_like(chunk_starts1, name, dtype=object)
        ))


        chunk_starts2 = np.arange(start2, end2, (chunk_size - overlap_size) * resolution)  # step = (chunk_size - overlap_size) * resolution
        chunk_ends2 = np.minimum(chunk_starts2 + chunk_size * resolution, end2)  # end = start + chunk_size * resolution


        chunks_id2 = list(zip(
            np.full_like(chunk_starts2, chrom2, dtype=object),
            chunk_starts2,
            chunk_ends2,
            np.full_like(chunk_starts2, name, dtype=object)
        ))


        if chrom1 == chrom2:
            # in cis
            for i in range(len(chunks_id1)):
                for j in range(i, len(chunks_id2)):
                    interacts_id.append((chunks_id1[i], chunks_id2[j]))
        else:
            # in trans
            for i in range(len(chunks_id1)):
                for j in range(len(chunks_id2)):
                    interacts_id.append((chunks_id1[i], chunks_id2[j]))        

    return clr, interacts_id



def idx_hic():

    return

