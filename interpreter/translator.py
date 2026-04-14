import bioframe
from matplotlib.colors import LogNorm
from multiprocessing import Pool
import time
import pandas as pd


class Translator:

    def __init__(self, **kwargs):
        self.check_configs(**kwargs)

    def check_configs(self, **kwargs):
        """
        check all the settings in the config
        """
        if "ref" not in kwargs:
            raise ValueError("Missing required parameter: ref")
        self.ref_type = kwargs['ref']


    def forward(self, region_li):

        ref_df = build_ref(self.ref_type)
        region_info_li, name_info_li = build_region(region_li, reference_df=ref_df)
        region_df = get_region_df(region_info_li, name_info_li)

        return region_df


def build_ref(token):

    if token == 'hg38':

        hg38_chromsizes = bioframe.fetch_chromsizes('hg38')
        hg38_cens = bioframe.fetch_centromeres('hg38')
        hg38_arms_full = bioframe.make_chromarms(hg38_chromsizes, hg38_cens)

        included_arms = hg38_arms_full["name"].to_list()[:44]
        hg38_arms = hg38_arms_full[hg38_arms_full["name"].isin(included_arms)].reset_index(drop=True)

        return hg38_arms

    else:

        return None


def build_region(region_li, reference_df):

    range_li = []
    meta_li = []

    for item in region_li:
        # ('range', 'name') or (('range', 'range'), 'name') or (('range', 'range'), ('name', 'name')) or (('range', 'range')) or ('range')
        if len(item) == 1:
            if len(item[0]) == 1:
                # ('range')
                range_li.append(item[0])
                name = region_search(item[0], lkup=reference_df)
                meta_li.append(name)

            elif len(item[0]) == 2:
                # (('range', 'range'))
                range_li.append(item[0])
                name1 = region_search(item[0][0], lkup=reference_df)
                name2 = region_search(item[0][1], lkup=reference_df)
                meta_li.append((name1, name2))
                                
            else:

                raise ValueError()

        elif len(item) == 2:
            if len(item[0]) == 1 and len(item[1]) == 1:
                # ('range', 'name')
                range_li.append(item[0])
                meta_li.append(item[1])

            elif len(item[0]) == 2 and len(item[1]) == 2:
                # (('range', 'range'), ('name', 'name'))
                range_li.append((item[0][0], item[0][1]))
                meta_li.append((item[1][0], item[1][1]))

            elif len(item[0]) == 2 and len(item[1]) == 1:
                # (('range', 'range'), 'name')
                range_li.append((item[0][0], item[0][1]))
                meta_li.append((item[1]))

            else:
                raise ValueError()
        
        else:

            raise ValueError()


    return range_li, meta_li



def region_search(string, lkup):

    if lkup is not None:

        chrom, start, end = str_to_info(string)

        mask = (
            (lkup["chrom"] == chrom) &
            (lkup["start"] <= start) &
            (lkup["end"] >= end)
        )

        match = lkup[mask]

        if match.shape[0] == 1:

            return match.iloc[0].name
        
        elif len(match) == 0:
            raise LookupError(f"{string} no matching items")

        else:

            raise LookupError("matching multiple items: found %d items" % match.shape[0])
        
    else:

        chrom, _, _ = str_to_info(string)

        return chrom


def str_to_info(string):
    
    chrom, range_part = string.split(':')
    start_str, end_str = range_part.split('-')

    start = int(start_str.replace(',', ''))
    end = int(end_str.replace(',', ''))

    return chrom, start, end


def name_merge(a, b):
    return "-".join(sorted([a, b]))


def get_region_df(region, metadata):

    data = []

    for item, name in zip(region, metadata):
        
        # if isinstance(item, str):
        if isinstance(item, tuple) and len(item) == 1:
            # cis
            chrom, start, end = str_to_info(item[0])

            if len(name) != 1:
                raise ValueError()
            
            name = name[0]
            data.append([chrom, start, end, chrom, start, end, name])

        elif isinstance(item, tuple) and len(item) == 2:
            # trans

            chrom1, start1, end1 = str_to_info(item[0])
            chrom2, start2, end2 = str_to_info(item[1])

            if len(name) == 2:
                name1, name2 = name
                name = name_merge(name1, name2)

            elif len(name) == 1:
                name = name[0]
            else:
                raise ValueError()

            data.append([chrom1, start1, end1, chrom2, start2, end2, name])

    df = pd.DataFrame(data, columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'name'])

    return df