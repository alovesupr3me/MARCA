import sys
import os
import numpy as np
import argparse

from interpreter.translator import Translator
from loader.indexer import Indexer
from filter.filter import Filter
from mc_caller.patch_caller import PatchCaller
from mc_caller.anchor_summarizer import AnchorSummarizer
import bioframe
from matplotlib.colors import LogNorm
from multiprocessing import Pool
import time
import itertools
import pandas as pd

import cooler
from normalizer.oe_norm_noz import OENormalizerContextNoneZero
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components



def process_wrapper_local(unique_key, clr, region_info, setting_dict):

    context_bp = setting_dict["norm_distance"]
    context_px = context_bp // int(setting_dict["resolution"])
    
    normer = OENormalizerContextNoneZero(context_px)
    model = Filter()
    caller = PatchCaller(**setting_dict)

    info_i = region_info[0]
    info_j = region_info[1]

    chrom_i, st_i, ed_i, nm_i = info_i
    chrom_j, st_j, ed_j, nm_j = info_j


    chrom_size_i, chrom_size_j = clr.chromsizes[chrom_i], clr.chromsizes[chrom_j]
    region_lkup = setting_dict["fetch_region"].set_index("name")

    region_size_i, region_size_j = region_lkup.loc[nm_i, "end1"], region_lkup.loc[nm_j, "end2"]

    chrom_size_i, chrom_size_j = min(chrom_size_i, region_size_i), min(chrom_size_j, region_size_j)

    fetch_st_i, fetch_ed_i = max(0, st_i - context_bp), min(chrom_size_i, ed_i + context_bp)
    fetch_st_j, fetch_ed_j = max(0, st_j - context_bp), min(chrom_size_j, ed_j + context_bp)

    key_i = f"{chrom_i}:{fetch_st_i}-{fetch_ed_i}"
    key_j = f"{chrom_j}:{fetch_st_j}-{fetch_ed_j}"

    observed = clr.matrix(balance=setting_dict["if_balanced"]).fetch(key_i, key_j)


    if st_i - context_bp < 0: observed = np.vstack([np.full((context_px, observed.shape[1]), np.nan), observed])
    if ed_i + context_bp >= chrom_size_i: observed = np.vstack([observed, np.full((context_px, observed.shape[1]), np.nan)])
    if st_j - context_bp < 0: observed = np.hstack([np.full((observed.shape[0], context_px), np.nan), observed])
    if ed_j + context_bp >= chrom_size_j: observed = np.hstack([observed, np.full((observed.shape[0], context_px), np.nan)])


    if_cis = (chrom_i == chrom_j)

    ob = observed[context_px:observed.shape[0] - context_px, context_px:observed.shape[1] - context_px]

    oe, m = normer.norm(observed, if_cis)
    m = m.astype(np.uint8)
    oe = np.nan_to_num(oe, copy=False, nan=0, posinf=0, neginf=0)

    output_wrapper = model.forward(oe)
    resp = output_wrapper["ellipse"]

    mcd_pixels = caller.forward(unique_key, region_info, ob, resp, m, if_cis)
 
    return mcd_pixels


def remove_dup_obj(all_mcd_pixels, all_mcd_boxes, all_mcd_pix_box):

    cols = ['chrom1','start1','end1', 'chrom2', 'start2', 'end2']
    mask = all_mcd_pix_box.duplicated(subset=cols, keep=False)  # keep=False 表示标记所有重复的行
    dup_pixels = all_mcd_pix_box[mask]

    if dup_pixels.empty:
        return all_mcd_pixels, all_mcd_boxes



    dup_pixels_agg = dup_pixels.groupby(cols)['region_id'].agg(list).reset_index(name='group_id')    


    exploded = dup_pixels_agg.explode('group_id').reset_index()  # index 保留原来的行
    edges_df = exploded.merge(exploded, left_on='index', right_on='index')
    edges_df = edges_df[edges_df['group_id_x'] != edges_df['group_id_y']]
    edges_df = edges_df[['group_id_x', 'group_id_y']].rename(
        columns={'group_id_x':'region_id_x','group_id_y':'region_id_y'}
    )
    edges_df = edges_df.drop_duplicates(subset=['region_id_x','region_id_y'])
    all_idx_grph = pd.unique(edges_df[['region_id_x', 'region_id_y']].values.ravel())
    reg_id_map = {rid: i for i, rid in enumerate(all_idx_grph)}
    id_reg_map = {i: rid for rid, i in reg_id_map.items()}
    edges_df['i'] = edges_df['region_id_x'].map(reg_id_map)
    edges_df['j'] = edges_df['region_id_y'].map(reg_id_map)

    row = edges_df['i'].to_numpy()
    col = edges_df['j'].to_numpy()
    data = np.ones(len(row), dtype=int)

    graph = csr_matrix((data, (row, col)), shape=(len(reg_id_map), len(reg_id_map)))

    n_components, labels = connected_components(csgraph=graph, directed=False)

    grp_li = [[] for _ in range(n_components)]
    for i, cci in enumerate(labels):
        grp_li[cci].append(id_reg_map[i])


    all_region_id_grp = [x for t in grp_li for x in t] 
    box_sampled = all_mcd_boxes[all_mcd_boxes['region_id'].isin(all_region_id_grp)]
    box_sampled = box_sampled.copy()
    box_sampled['area'] = box_sampled['bin1_width'] * box_sampled['bin2_width']


    grp_df = pd.DataFrame({
        'idx': range(len(grp_li)),
        'groups': grp_li
    }).explode('groups')

    grp_df = grp_df.merge(
        box_sampled[['region_id', 'area']],
        left_on='groups',
        right_on='region_id',
        how='left'
    ).drop(columns='region_id').rename(columns={'groups': 'region_id_in_group'})

    max_id_grp = grp_df.groupby('idx')['area'].idxmax()
    kept_id_grp = grp_df.loc[max_id_grp, 'region_id_in_group'].tolist()
    id_remove_grp = list(set(all_region_id_grp) - set(kept_id_grp))

    all_mcd_pixels = all_mcd_pixels[~all_mcd_pixels['region_id'].isin(id_remove_grp)]
    all_mcd_boxes = all_mcd_boxes[~all_mcd_boxes['region_id'].isin(id_remove_grp)]


    return all_mcd_pixels, all_mcd_boxes


if __name__ == "__main__":
    
    ###### temporally testing stage
    # HiC-cis
    
    dataset = "HiC"    
    hg38_chromsizes = bioframe.fetch_chromsizes('hg38')
    hg38_cens = bioframe.fetch_centromeres('hg38')
    hg38_arms_full = bioframe.make_chromarms(hg38_chromsizes, hg38_cens)

    included_arms = hg38_arms_full["name"].to_list()[:44]
    hg38_arms = hg38_arms_full[hg38_arms_full["name"].isin(included_arms)].reset_index(drop=True)
    regions = [((f"{row.chrom}:{row.start:,}-{row.end:,}",), (row['name'],)) for _, row in hg38_arms.iterrows()]


    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--filename", type=str)
    parser.add_argument("--n_proc", type=int, default=16)
    parser.add_argument("--resolution", type=str, default="10000")
    parser.add_argument("--norm_pix", type=int, default=200)
    parser.add_argument("--peak_thr", type=float, default=0.25)
    parser.add_argument("--bottom_thr", type=float, default=0.03)
    parser.add_argument("--caller_resolution", type=float, default=0.005)
    parser.add_argument("--filtered_zscore", type=float, default=0.8)
    parser.add_argument("--range_thr", type=float, default=0.1)
    parser.add_argument("--nan_thr", type=float, default=0.1)
    parser.add_argument("--reach_thr", type=int, default=4000000)
    parser.add_argument("--agg_resp_thr", type=int, default=0.1)
    parser.add_argument("--conn_conf_thr", type=int, default=0.1)     

    # parser.add_argument("--norm_distance", type=int, default="200000")

    args = parser.parse_args()

    config = {
        "filename": args.filename,
        "outdir": args.outdir,
        "resolution": args.resolution,
        "if_balanced": True,
        "fetch_region": regions,
        # "norm_distance": 2_000_000,
        "norm_distance": args.norm_pix * int(args.resolution),
        "chunk_size": 384,
        "filtered_zscore": args.filtered_zscore,
        "overlap_size": 20,
        "n_proc": args.n_proc,
        "peak_thr": args.peak_thr,
        "bottom_thr": args.bottom_thr,
        "caller_resolution": args.caller_resolution,
        "range_thr": args.range_thr,
        "nan_thr": args.nan_thr,
        "reach_thr": args.reach_thr,
        "agg_resp_thr": args.agg_resp_thr,
        "conn_conf_thr": args.conn_conf_thr,  
        "ref": None  # testing
    }

    print("MARCA configs:", config)


    print("Generate fetch region..")

    tr = Translator(**config)
    config["fetch_region"] = tr.forward(config["fetch_region"])

    print("Index all patches' urls of the file..")
    idxr = Indexer(**config)
    clr, ids = idxr.index()

    start = time.time()


    print("Start multiprocessing..")

    with Pool(config["n_proc"]) as pool:
        results = pool.starmap(process_wrapper_local, [(str(i), clr, ids[i], config) for i in range(len(ids))])
    

    end = time.time()
    print(f"time: {end - start:.3f} sec")

    mcd_pixel_list, mcd_box_list, mcd_pix_box_list = zip(*results)

    all_mcd_pixels = pd.concat(mcd_pixel_list, ignore_index=True)
    all_mcd_boxes = pd.concat(mcd_box_list, ignore_index=True)
    all_mcd_pix_box = pd.concat(mcd_pix_box_list, ignore_index=True)

    
    print("Remove patch-level overlapping pixels..")
    all_mcd_pixels, all_mcd_boxes = remove_dup_obj(all_mcd_pixels, all_mcd_boxes, all_mcd_pix_box)


    print("Filtering and Summerizing in region-level..")


    grouped_mcd_pixels = all_mcd_pixels.groupby("name")
    grouped_mcd_boxes = all_mcd_boxes.groupby("name")


    group_names = set(grouped_mcd_pixels.groups) & set(grouped_mcd_boxes.groups)


    detected_mcd_boxes = []
    detected_mcd_pixels = []

    all_anchors = []
    

    for gname in group_names:


        print(f"dealing with {gname}..")


        row = config["fetch_region"].loc[config["fetch_region"]["name"] == gname].iloc[0]
        region_info = {"chrom1": row.chrom1, "start1": int(row.start1), "end1": int(row.end1), "chrom2": row.chrom2, "start2": int(row.start2), "end2": int(row.end2)}
        if_trans = row["chrom1"] != row["chrom2"]

        post_caller = AnchorSummarizer(**config)

        sub_mcd_pixels = grouped_mcd_pixels.get_group(gname)
        sub_mcd_boxes = grouped_mcd_boxes.get_group(gname)

        grouped_mcd_pixels_filtered, grouped_mcd_boxes_filtered, anchors = post_caller.forward(sub_mcd_pixels, sub_mcd_boxes, region_info, if_trans)

        # anchor summarizing & detection recticfying 

        anchors.insert(0, "region", gname)

        all_anchors.append(anchors)
        detected_mcd_boxes.append(grouped_mcd_boxes_filtered)
        detected_mcd_pixels.append(grouped_mcd_pixels_filtered)



    mcd_anchors = pd.concat(all_anchors, ignore_index=True) if all_anchors else pd.DataFrame()
    mcd_boxes = pd.concat(detected_mcd_boxes, ignore_index=True) if detected_mcd_boxes else pd.DataFrame()
    mcd_pixels = pd.concat(detected_mcd_pixels, ignore_index=True) if detected_mcd_pixels else pd.DataFrame()


    print("Save all results..")

    dirname = config["outdir"]
    os.makedirs(dirname, exist_ok=True)  
  
    if not mcd_boxes.empty:
        mcd_boxes.insert(0, "region", mcd_boxes["name"])
        save_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "region_id", "region", "intensity", "if_nan"]
        fname = f"{dirname}/mcd_boxes.bedpe"
        mcd_boxes[save_cols].to_csv(
            fname,
            index=False,
            sep="\t"
        )

    if not mcd_anchors.empty:
        save_cols = ["chrom", "start", "end", "region", "anchor_id"]
        fname = f"{dirname}/mcd_anchors.bed"
        mcd_anchors[save_cols].to_csv(
            fname,
            index=False,
            sep="\t"
        )

    print("Finished.")

    end = time.time()
    print(f"time: {end - start:.3f} sec")
