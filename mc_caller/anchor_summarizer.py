import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import bioframe
from itertools import combinations
from scipy.sparse import csr_matrix
import networkx as nx
import time
import random
from collections import defaultdict


class AnchorSummarizer:

    def __init__(self, **kwargs):
        self.check_configs(**kwargs)


    def check_configs(self, **kwargs):
            """
            check all the settings in the config
            """
            if "resolution" not in kwargs:
                raise ValueError("Missing required parameter: resolution")        
            self.res = int(kwargs["resolution"])
            if "range_thr" not in kwargs:
                raise ValueError("Missing required parameter: range_thr")        
            self.range_th = kwargs["range_thr"] 
            if "nan_thr" not in kwargs:
                raise ValueError("Missing required parameter: nan_thr")        
            self.nan_th = kwargs["nan_thr"]
            if "reach_thr" not in kwargs:
                raise ValueError("Missing required parameter: reach_thr")        
            self.reach_th = kwargs["reach_thr"]  
            if "agg_resp_thr" not in kwargs:
                raise ValueError("Missing required parameter: agg_resp_thr")        
            self.AR = kwargs["agg_resp_thr"]                      
            if "conn_conf_thr" not in kwargs:
                raise ValueError("Missing required parameter: conn_conf_thr")        
            self.CC = kwargs["conn_conf_thr"]  

    def forward(self, region_mcd_pixels, all_region_mcd_boxes, region_info, if_trans):

        all_region_mcd_boxes = diag_projection_df(all_region_mcd_boxes)

        region_nan_boxes = all_region_mcd_boxes[all_region_mcd_boxes["if_nan"]]
        region_mcd_boxes = all_region_mcd_boxes[~all_region_mcd_boxes["if_nan"]]

        region_nan_boxes = region_nan_boxes[region_nan_boxes["intensity"] >= self.nan_th]

        if if_trans:
            
            # do not caputre spatial relationship on trans-region
            all_return = pd.concat([region_mcd_boxes, region_nan_boxes], ignore_index=True)
            region_ids = all_return["region_id"].unique()

            pix_msk = region_mcd_pixels["region_id"].isin(region_ids)
            box_msk = all_region_mcd_boxes["region_id"].isin(region_ids)

            return region_mcd_pixels[pix_msk], all_region_mcd_boxes[box_msk], pd.DataFrame()


        mcd_boxes_dict = dict(tuple(region_mcd_boxes.groupby('intensity')))

        th_idx =  np.array(sorted(region_mcd_boxes['intensity'].unique(), reverse=True))
        high_group = th_idx[th_idx >= self.range_th].tolist()
        low_group = th_idx[th_idx < self.range_th].tolist()


        mcd_np = np.empty((0, 4))
        N = []

        for th in high_group:
            S = mcd_boxes_dict[th]
            mcd_add = transfer(S)
            mcd_np = np.vstack([mcd_np, mcd_add])

            N.extend(S['region_id'].tolist())


        W_np = np.empty((0, 4))
        W_df = region_mcd_boxes.head(0).copy()
        C = np.empty((0, 2))
        C_grp_li = []


        R_grp_li = link(mcd_np)
        R = project_ini(mcd_np, R_grp_li, self.res)
        

        for th in low_group:


            S_df = mcd_boxes_dict[th]
            S_np = transfer(S_df)
            

            R_anno_np, R_slct_np = activate(S_np, R, self.res)
            S_acti_by_R, S_acti_by_R_idx = search(S_np, R_slct_np, R, self.res)

            N.extend(S_df.iloc[S_acti_by_R_idx]['region_id'].tolist())
            S_np = np_mask(S_np, R_anno_np)
            

            C_anno_np, C_slct_np = activate(S_np, C, self.res)
            S_acti_by_C, S_acti_by_C_idx = search(S_np, C_slct_np, C, self.res)
            W_acti_by_C, W_acti_by_C_idx = search(W_np, C_slct_np, C, self.res)
            

            N.extend(S_df.iloc[S_acti_by_C_idx]['region_id'].tolist())
            N.extend(W_df.iloc[W_acti_by_C_idx]['region_id'].tolist())

            W_df = W_df.drop(W_df.index[W_acti_by_C_idx])

            S_np = np_mask(S_np, C_anno_np)
            R = np.vstack([R, C[C_slct_np]])


            Sp_fake_anno_np, Sp_grp_li, S_acti_by_Sp_idx = find_n_link(S_np)
            Sp = project(S_np, Sp_grp_li, self.res)
            S_np = np_mask(S_np, Sp_fake_anno_np)
            N.extend(S_df.iloc[S_acti_by_Sp_idx]['region_id'].tolist())
            R = np.vstack([R, Sp])


            S_left, S_left_id = compress(S_np)

            S_left_df = S_df.iloc[S_left_id]
            W_df = pd.concat([W_df, S_left_df], ignore_index=True)
            

            W_np = transfer(W_df)
            C_grp_li = link(W_np) 
            C = project(W_np, C_grp_li, self.res)


            C_kp = note_non_overlap(R, C)
            C = C[C_kp]


        S_nan_np = transfer(region_nan_boxes)
        _, R_slct_np = activate(S_nan_np, R, self.res)
        _, S_nan_acti_by_R_idx = search(S_nan_np, R_slct_np, R, self.res)
        
        all_acti_nan_region_id = set(region_nan_boxes.iloc[S_nan_acti_by_R_idx]['region_id'].tolist())

        N.extend(region_nan_boxes.iloc[S_nan_acti_by_R_idx]['region_id'].tolist())


        N = set(N)
        all_region_mcd_boxes = filter_down_diag(all_region_mcd_boxes)
        
        pix_msk = region_mcd_pixels["region_id"].isin(N)
        box_msk = all_region_mcd_boxes["region_id"].isin(N)

        new_region_mcd_pixels = region_mcd_pixels[pix_msk]
        new_region_mcd_boxes = all_region_mcd_boxes[box_msk]


        chrn = all_region_mcd_boxes['chrom1'].unique()
        chrn = str(chrn[0])
        anc = gen_anc(R, self.res, chrn)



        nan_msk = all_region_mcd_boxes["region_id"].isin(all_acti_nan_region_id)
        acti_nan_box = all_region_mcd_boxes[nan_msk]
        nan_box_x = acti_nan_box[["chrom1", "start1", "end1", "region_id"]].rename(
            columns={"chrom1":"chrom", "start1":"start", "end1":"end"}
        )
        nan_box_y = acti_nan_box[["chrom2", "start2", "end2", "region_id"]].rename(
            columns={"chrom2":"chrom", "start2":"start", "end2":"end"}
        )

        nan_anc = pd.concat([nan_box_x, nan_box_y])
        temp_anc = pd.concat([anc, nan_anc])
        temp_anc = bioframe.merge(temp_anc)
        

        ref_len = min(abs(region_info["start1"] - region_info["end1"]), abs(region_info["start2"] - region_info["end2"]))
        examine_box_df = pd.concat([region_mcd_boxes, region_nan_boxes], ignore_index=True)
        temp_anc_w_info = find_box_for_each_anchor(temp_anc, examine_box_df)
        temp_anc_w_info = anc_max_d(temp_anc_w_info, examine_box_df)
        temp_anc_w_info["max_d_per"] = temp_anc_w_info["max_d"] / ref_len
        temp_anc = temp_anc_w_info[temp_anc_w_info["max_d"] >= self.reach_th].reset_index(drop=True)

        if len(temp_anc) == 0:
            return region_mcd_pixels.head(0), all_region_mcd_boxes.head(0), pd.DataFrame()
        

        proj1 = new_region_mcd_boxes[["chrom1","start1","end1","region_id", "intensity"]].rename(
        columns={"chrom1":"chrom", "start1":"start", "end1":"end"}
        )
        proj2 = new_region_mcd_boxes[["chrom2","start2","end2","region_id", "intensity"]].rename(
            columns={"chrom2":"chrom", "start2":"start", "end2":"end"}
        )

        ovlp = bioframe.overlap(pd.concat([proj1, proj2]), temp_anc)

        ovlp = ovlp.dropna(subset=['chrom_','start_','end_']).reset_index(drop=True)
        candi_i_agg = ovlp.groupby(['chrom_', 'start_', 'end_'])['intensity'].sum().reset_index()
        
        # filtering
        temp_anc = candi_i_agg[candi_i_agg["intensity"] >= self.AR].reset_index(drop=True)
        temp_anc = temp_anc.rename(columns={"chrom_": "chrom", "start_": "start", "end_": "end"})


        candi_anc = temp_anc.reset_index().rename(columns={"index":"anchor_id"})
        proj1 = all_region_mcd_boxes[["chrom1","start1","end1","region_id", "intensity"]].rename(
        columns={"chrom1":"chrom", "start1":"start", "end1":"end"}
        )
        proj2 = all_region_mcd_boxes[["chrom2","start2","end2","region_id", "intensity"]].rename(
            columns={"chrom2":"chrom", "start2":"start", "end2":"end"}
        )


        if len(candi_anc) == 0:
            return region_mcd_pixels.head(0), all_region_mcd_boxes.head(0), pd.DataFrame()
        

        ovlp = bioframe.overlap(pd.concat([proj1, proj2]), candi_anc)

        ovlp = ovlp.dropna(subset=['chrom_','start_','end_']).reset_index(drop=True)
        overlap_dict = ovlp.groupby("region_id")["anchor_id_"].apply(list).to_dict()


        row, col, data = [], [], []
        node_degree = defaultdict(int) 
    
        for rid, anc_list in overlap_dict.items():
            if len(anc_list) < 2:
                continue

            weight = all_region_mcd_boxes.loc[
                all_region_mcd_boxes["region_id"] == rid, "intensity"
            ].max()

            for a, b in combinations(anc_list, 2):

                row.extend([int(a), int(b)])
                col.extend([int(b), int(a)])
                data.extend([weight, weight])

                node_degree[int(a)] += 1
                node_degree[int(b)] += 1

        n_nodes = len(candi_anc)
        graph = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))


        gscores = cal_2hop_support(graph)
        selected_anc_ids = set(np.where(gscores >= self.CC)[0])
        anc_regions= candi_anc[candi_anc["anchor_id"].isin(selected_anc_ids)]

        new_region_mcd_pixels, new_region_mcd_boxes = re_filter(new_region_mcd_pixels, new_region_mcd_boxes, anc_regions)

        return new_region_mcd_pixels, new_region_mcd_boxes, anc_regions


def cal_2hop_support(A):

    A.setdiag(0)
    A.eliminate_zeros()

    n = A.shape[0]
    ones = np.ones(n, dtype=float)
    t = A.dot(ones)

    At = A.dot(t)


    A_sq = A.multiply(A)
    row_sq_sum = np.array(A_sq.sum(axis=1)).ravel()

    S = At - row_sq_sum

    return S


def re_filter(mcd_pixels, mcd_boxes, mcd_anchors):

    if mcd_anchors is None or len(mcd_anchors) == 0:
        return mcd_pixels.head(0), mcd_boxes.head(0)

    box_x = mcd_boxes[["chrom1", "start1", "end1", "region_id"]].rename(
        columns={"chrom1":"chrom", "start1":"start", "end1":"end"}
    )
    box_y = mcd_boxes[["chrom2", "start2", "end2", "region_id"]].rename(
        columns={"chrom2":"chrom", "start2":"start", "end2":"end"}
    )
    
    ovlp = bioframe.overlap(pd.concat([box_x, box_y]), mcd_anchors)
    ovlp = ovlp.dropna(subset=['chrom_','start_','end_']).reset_index(drop=True)

    valid_ids = ovlp["region_id"].unique()

    pixel_filtered = mcd_pixels[mcd_pixels["region_id"].isin(valid_ids)].reset_index(drop=True)
    box_filtered = mcd_boxes[mcd_boxes["region_id"].isin(valid_ids)].reset_index(drop=True)

    return pixel_filtered, box_filtered


def find_n_link(loc_mx):

    mask = ~np.all(loc_mx == -1, axis=1)
    if mask.sum() <= 1:
        return np.full((loc_mx.shape[0], 1), -1, dtype=int), [], np.array([], dtype=int)

    idxs = np.nonzero(mask)[0]
    starts = loc_mx[mask, 0].astype(float)
    ends = loc_mx[mask, 1].astype(float)

    order = np.lexsort(( -ends, starts ))

    groups = {}
    max_end = -np.inf
    max_orig = None

    for oi in order:
        orig = idxs[oi]
        e = ends[oi]
        if e <= max_end:

            groups.setdefault(max_orig, [max_orig]).append(orig)
        else:

            max_end = e
            max_orig = orig
            groups.setdefault(max_orig, [max_orig])


    grp_li = [np.array(v, dtype=int) for v in groups.values() if len(v) > 1]

    fake_anno = np.full((loc_mx.shape[0], 1), -1, dtype=int)
    
    all_acti_id = np.unique(np.concatenate(grp_li)) if grp_li else np.array([], int)
    
    fake_anno[all_acti_id, :] = 0

    return fake_anno, grp_li, all_acti_id




def transfer(loc_df):

    loc_np = loc_df[['start1', 'end1', 'start2', 'end2']].to_numpy()

    return loc_np


def link(loc_np):

    loc_np_p = loc_np[:, 0:2]

    if len(loc_np_p) == 0:
        return []
    
    idx_li = group_overlapping_obj(loc_np_p)

    return idx_li


def project_ini(loc_np, idx_li, res):

    loc_np_p = loc_np[:, 0:2]

    intv_np = obj_projection_ini(loc_np_p, idx_li, resolution=res)

    return intv_np


def project(loc_np, idx_li, res):

    loc_np_p = loc_np[:, 0:2]
    
    intv_np = obj_projection(loc_np_p, idx_li, resolution=res)

    return intv_np


def activate(loc_np, intv_np, resolution):

    loc_np_p = loc_np[:, 0:2]

    anno_np, slct_np = interval_match(loc_np_p, intv_np, resolution)

    return anno_np, slct_np


def np_mask(loc_np, anno_np):

    idx = np.where(~np.all(anno_np == -1, axis=1))[0]
    loc_np[idx, :] = -1

    return loc_np


def search(loc_np, slct_np, intv_np, resolution):

    acti_intv_np = intv_np[slct_np]

    anno_np, _ = interval_match(loc_np, acti_intv_np, resolution)

    idx_all = np.where(~np.all(anno_np == -1, axis=1))[0]

    return loc_np[idx_all], idx_all


def compress(loc_np):

    val_idx = np.where(~np.all(loc_np == -1, axis=1))[0]

    val_loc_np = loc_np[val_idx]
    return val_loc_np, val_idx


def group_overlapping_obj(intervals):

    idx_sort = np.argsort(intervals[:, 0])
    intervals_sorted = intervals[idx_sort]

    groups = []
    current_group = [idx_sort[0]]
    current_end = intervals_sorted[0, 1]

    for i in range(1, len(intervals_sorted)):
        start, end = intervals_sorted[i]
        if start <= current_end:
            current_group.append(idx_sort[i])
            current_end = max(current_end, end)
        else:
            groups.append(current_group)
            current_group = [idx_sort[i]]
            current_end = end

    groups.append(np.array(current_group))
    return groups


def interval_match(loc_mx, intv_mx, resolution):

    loc_st = loc_mx[:, 0]
    loc_ed = loc_mx[:, 1]
    intv_st = intv_mx[:, 0]
    intv_ed = intv_mx[:, 1]

    cond_start = intv_st[None, :] <= loc_st[:, None]
    cond_end   = loc_ed[:, None] <= intv_ed[None, :]
    inside = cond_start & cond_end

    if inside.shape[1] == 0:
        return np.full((loc_mx.shape[0], 1), -1, dtype=int), np.array([], dtype=int)


    matched_loc_ids = inside.astype(int).argmax(axis=1)

    matched_loc_ids[~inside.any(axis=1)] = -1

    matched_intv_ids = np.unique(matched_loc_ids[matched_loc_ids != -1])

    return matched_loc_ids.reshape(-1, 1), matched_intv_ids


def obj_projection(loc, grp_list, resolution):
    cen = (loc[:, 0] + loc[:, 1]) / 2
    diam = (loc[:, 1] - loc[:, 0])

    res = np.zeros((len(grp_list), 2))

    for idx, grp in enumerate(grp_list):

        sel_cen = cen[grp]
        sel_d = diam[grp]
        
        c = weighted_median(sel_cen, sel_d)
        r = np.average(sel_d) / 2
        st = round(c - r)
        ed = round(c + r)

        st = st // resolution * resolution
        ed = math.ceil(ed / resolution) * resolution


        res[idx, :] = np.array([st, ed])

    return res


def weighted_median(data, weights):

    sorted_idx = np.argsort(data)
    data, weights = data[sorted_idx], weights[sorted_idx]
    cum_weights = np.cumsum(weights)
    cutoff = 0.5 * np.sum(weights)
    return data[np.searchsorted(cum_weights, cutoff)]


def note_non_overlap(R_intv, C_intv):

    x_ovlp_idx = label_overlap_anchor(R_intv, C_intv)
    C_x_idx = np.arange(C_intv.shape[0])
    C_x_kp_idx = np.setdiff1d(C_x_idx, x_ovlp_idx)


    return C_x_kp_idx


def label_overlap_anchor(R_intv_mx, C_intv_mx):

    ovlp_in_C = (C_intv_mx[:, None, 0] < R_intv_mx[None, :, 1]) & (R_intv_mx[None, :, 0] < C_intv_mx[:, None, 1])

    C_ovlp_idx = np.where(ovlp_in_C.any(axis=1))[0]

    return np.unique(C_ovlp_idx)


def gen_anc(R, resolution, chrname):

    x_st = R[:, 0]
    x_ed = R[:, 1]

    x_df = pd.DataFrame({
        "chrom": chrname,
        "start": (x_st // resolution * resolution).astype(int),
        "end": np.ceil(x_ed / resolution).astype(int) * resolution,
    })

    return x_df


def diag_projection_df(df):
    df_transposed = df.rename(columns={
        "chrom1": "chrom2",
        "start1": "start2",
        "end1": "end2",
        "chrom2": "chrom1",
        "start2": "start1",
        "end2": "end1"
    })

    df_new = pd.concat([df, df_transposed], ignore_index=True)

    return df_new


def filter_down_diag(box_df):

    filtered_box = box_df[(box_df["start1"] >= box_df["start2"]) | (box_df["end1"] >= box_df["end2"])]

    filtered_box_df = box_df.drop(filtered_box.index).reset_index(drop=True)

    return filtered_box_df


def obj_projection_ini(loc, grp_list, resolution):
    st_v = loc[:, 0]
    ed_v = loc[:, 1]

    res = np.zeros((len(grp_list), 2))

    for idx, grp in enumerate(grp_list):

        sel_st = st_v[grp]
        sel_ed = ed_v[grp]
        
        st = sel_st.min()
        ed = sel_ed.max()
        res[idx, :] = np.array([st, ed])

    return res


def find_box_for_each_anchor(anc_df, box_df):

    proj1 = box_df.rename(columns={"chrom1": "chrom", "start1": "start", "end1": "end"})
    proj2 = box_df.rename(columns={"chrom2": "chrom", "start2": "start", "end2": "end"})
    box_proj_df = pd.concat([proj1, proj2], ignore_index=True)[["chrom", "start", "end", "region_id"]]

    overlap_df = bioframe.overlap(
        anc_df,
        box_proj_df,
        how="left",
        suffixes=("_anc", "_box")
    )

    grouped = (
        overlap_df.dropna(subset=["region_id_box"])
        .groupby(["chrom_anc", "start_anc", "end_anc"])["region_id_box"]
        .agg(lambda x: list(set(x)))
        .reset_index()
    )


    anc_df_w_region = anc_df.merge(
        grouped,
        how="left",
        left_on=["chrom", "start", "end"],
        right_on=["chrom_anc", "start_anc", "end_anc"]
    ).drop(columns=["chrom_anc", "start_anc", "end_anc"])


    anc_df_w_region = anc_df_w_region.rename(columns={"region_id_box": "matched_region_id"})
    return anc_df_w_region


def anc_max_d(anc_df_w_region, box_df):

    box_df = box_df.copy()
    box_df["center1"] = (box_df["start1"] + box_df["end1"]) / 2
    box_df["center2"] = (box_df["start2"] + box_df["end2"]) / 2
    box_df["d"] = np.abs(box_df["center2"] - box_df["center1"])
    
    box_d = box_df.set_index("region_id")["d"].to_dict()
    
    def get_max_d(region_ids):
        if not isinstance(region_ids, list) or len(region_ids) == 0:
            return np.nan
        d_vals = [box_d[rid] for rid in region_ids if rid in box_d]
        return np.max(d_vals) if d_vals else np.nan

    anc_df_w_region = anc_df_w_region.copy()
    anc_df_w_region["max_d"] = anc_df_w_region["matched_region_id"].apply(get_max_d)

    return anc_df_w_region