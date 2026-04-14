import numpy as np
from scipy.ndimage import label as scipy_label
from skimage.measure import label as skimage_label, regionprops
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_dilation
from skimage.morphology import reconstruction
import pandas as pd
from skimage.morphology import remove_small_objects
import imageio



class PatchCaller:

    def __init__(self, **kwargs):
        self.check_configs(**kwargs)
        self.min_area = 5
        self.nan_d = 5
        self.if_catch_nan = True


    def check_configs(self, **kwargs):
            """
            check all the settings in the config
            """
            if "resolution" not in kwargs:
                raise ValueError("Missing required parameter: resolution")        
            self.res = int(kwargs["resolution"])
            if "chunk_size" not in kwargs:
                raise ValueError("Missing required parameter: chunk_size")        
            self.chunk_size = kwargs["chunk_size"]
            if "filtered_zscore" not in kwargs:
                raise ValueError("Missing required parameter: filtered_zscore: min zscore for detected microcompartment interaction")        
            self.min_z = kwargs["filtered_zscore"]
            if "peak_thr" not in kwargs:
                raise ValueError("Missing required parameter: peak_thr")        
            self.max_th = kwargs["peak_thr"]
            if "bottom_thr" not in kwargs:
                raise ValueError("Missing required parameter: bottom_thr")        
            self.min_th = kwargs["bottom_thr"]
            if "caller_resolution" not in kwargs:
                raise ValueError("Missing required parameter: caller_resolution")        
            self.step = kwargs["caller_resolution"]


    def forward(self, unique_key, chunk_info, raw, resp, nan, if_cis):

        bottom = np.where(resp > self.min_th, 1, 0)
        
        bottom_labeled, _ = scipy_label(bottom, structure=np.ones((3, 3)))


        thr_list = threshold_view(self.max_th, self.min_th, self.step)
        
        dec_mask = np.ones(resp.shape, dtype=bool)
        multi_note_map = np.zeros(resp.shape, dtype=np.float32)
        multi_thr_map  = np.zeros(resp.shape, dtype=np.float32)
        slic_t =  np.zeros(resp.shape, dtype=bool)

        
        for th in thr_list:
            slic_t_1 = slic_t

            slic_t = (resp >= th)
            slic_t = remove_small_objects(slic_t, min_size=self.min_area)

            marker_u8 = slic_t_1.astype(np.uint8)
            mask_u8   = slic_t.astype(np.uint8)
            slic_t_1 = reconstruction(marker_u8, mask_u8, method='dilation').astype(bool)

            new_pixels = slic_t & (~slic_t_1)
            new_obj_within  = new_pixels & (~dec_mask)
            new_obj_without = new_pixels & dec_mask


            slic_filtered_labeled = new_obj_without.astype(np.int64) * bottom_labeled
            slic_filtered_ids = np.unique(slic_filtered_labeled)
            slic_filtered_ids = slic_filtered_ids[slic_filtered_ids != 0]


            broad_to_bottom = np.isin(bottom_labeled, slic_filtered_ids)


            dec_mask = dec_mask & (~broad_to_bottom)


            multi_note_map = np.where(broad_to_bottom, th, multi_note_map)
            multi_note_map = np.where(new_obj_within, th, multi_note_map)

            multi_thr_map  = np.where(broad_to_bottom, th, multi_thr_map)
            multi_thr_map  = np.where(new_obj_within, th, multi_thr_map)


        seg = ((resp >= multi_thr_map) & (multi_thr_map > 0)).astype(float)



        seg_filtered, seg_nan, seg_bord = neglect_mask(seg, nan, distance=self.nan_d)

        if_catch_nan = self.if_catch_nan

        if if_catch_nan:
           
            acti = scaled_zf(raw, seg_filtered, (seg_filtered * multi_note_map), self.chunk_size, self.min_z)

            seg_filtered = acti * seg_filtered

            seg = seg_filtered + seg_nan

        else:

            acti = scaled_zf(raw, seg_filtered, (seg_filtered * multi_note_map), self.chunk_size, self.min_z)

            seg = acti * seg_filtered

        seg_info_map = seg * multi_note_map

        seg_labeled, _ = scipy_label(seg, structure=np.ones((3, 3)))
        seg_props = regionprops(seg_labeled, intensity_image=seg_info_map)

        nan_labels = seg_labeled * seg_nan
        nan_obj_ids = np.unique(nan_labels)
        nan_obj_ids = nan_obj_ids[nan_obj_ids != 0]
        nan_obj_ids = set(nan_obj_ids)


        all_pixels_df, all_boxes_df, all_pix_in_boxes = generate_pix_and_box_df(self.res, unique_key, (seg_labeled, seg_props, nan_obj_ids), chunk_info)

        if if_cis:
            all_pixels_df, all_boxes_df, all_pix_in_boxes = filter_down_diag(all_pixels_df, all_boxes_df, all_pix_in_boxes)


        return all_pixels_df, all_boxes_df, all_pix_in_boxes


def neglect_mask(seg, nan, distance=5):

    dis = 2 * distance + 1
    nan_mask1 = binary_dilation(nan, structure=np.ones((dis, dis)))

    H, W = nan.shape
    nan_mask2 = np.zeros_like(nan)
    nan_mask2[0:1, :] = 1
    nan_mask2[:, 0:1] = 1
    nan_mask2[:, W-1:W] = 1
    nan_mask2[H-1:H, :] = 1


    label, _ = scipy_label(seg, structure=np.ones((3, 3)))

    selected1 = label[nan_mask1 > 0]
    selected1 = np.unique(selected1)
    selected1 = selected1[selected1 != 0]  

    selected2 = label[nan_mask2 > 0]
    selected2 = np.unique(selected2)
    selected2 = selected2[selected2 != 0]  

    unique_selected1 = np.setdiff1d(selected1, selected2)
    union_selected = np.union1d(selected1, selected2)

    selected_near_nan = np.isin(label, unique_selected1)
    selected_to_mask = np.isin(label, union_selected)

    edge_objs = np.isin(label, selected2)

    return ((seg.astype(bool)) & (~selected_to_mask)).astype(np.uint8), ((selected_near_nan)).astype(np.uint8), edge_objs.astype(np.uint8)


def threshold_view(max_th, min_th, step):

    record = np.arange(max_th, min_th - 1e-10, -step)
    record = np.round(record, 5)

    return record.tolist()


def generate_pix_and_box_df(res, key, seg_wrapper, chunk_info):

    seg_labeled = seg_wrapper[0]
    seg_info = seg_wrapper[1]
    nan_obj_id_set = seg_wrapper[2]


    info_i = chunk_info[0]
    info_j = chunk_info[1]

    if info_i[3] != info_j[3]:
        print("patch on 2 regions")

    name = info_i[3]

    coords = np.argwhere(seg_labeled > 0)
    region_ids = seg_labeled[coords[:, 0], coords[:, 1]]
    region_id_str = np.char.add(key + "_", region_ids.astype(str))

    coords_bp = coords * res + np.array([[info_i[1], info_j[1]]])
    i = coords_bp[:, 0]
    j = coords_bp[:, 1]

    all_pixels_df = pd.DataFrame({
        "chrom1": info_i[0],
        "start1": i,
        "end1": i + res,
        "chrom2": info_j[0],
        "start2": j,
        "end2": j + res,
        "region_id": region_id_str,
        "name": name,
    })

    all_boxes_df = pd.DataFrame({
        "chrom1": pd.Series(dtype="object"),
        "start1": pd.Series(dtype="int64"),
        "end1": pd.Series(dtype="int64"),
        "chrom2": pd.Series(dtype="object"),
        "start2": pd.Series(dtype="int64"),
        "end2": pd.Series(dtype="int64"),
        "region_id": pd.Series(dtype="object"),
        "bin1_width": pd.Series(dtype="int64"),
        "bin2_width": pd.Series(dtype="int64"),
        "name": pd.Series(dtype="object"),
        "intensity": pd.Series(dtype="float64"),
        "if_nan": pd.Series(dtype="bool"),
    })

    all_pix_from_boxes = pd.DataFrame({
        "chrom1": pd.Series(dtype="object"),
        "start1": pd.Series(dtype="int64"),
        "end1": pd.Series(dtype="int64"),
        "chrom2": pd.Series(dtype="object"),
        "start2": pd.Series(dtype="int64"),
        "end2": pd.Series(dtype="int64"),
        "region_id": pd.Series(dtype="object"),
        "name": pd.Series(dtype="object"),
    })

    rows = []
    pix_rows = []
    i_offset = info_i[1]
    j_offset = info_j[1]

    for meta_region in seg_info:
        minr, minc, maxr, maxc = meta_region.bbox
        key_region_id = key + "_" + str(meta_region.label)

        rows.append({
            "chrom1": info_i[0],
            "start1": minr * res + i_offset,
            "end1": maxr * res + i_offset,
            "chrom2": info_j[0],    
            "start2": minc * res + j_offset,
            "end2": maxc * res + j_offset,
            "region_id": key_region_id,
            "bin1_width": maxr - minr,
            "bin2_width": maxc - minc,
            "name": name,
            "intensity": round(meta_region.mean_intensity, 5),
            "if_nan": meta_region.label in nan_obj_id_set
        })


        rows_coord = np.arange(minr * res + i_offset, maxr * res + i_offset, res)
        cols_coord = np.arange(minc * res + j_offset, maxc * res + j_offset, res)
        rr, cc = np.meshgrid(rows_coord, cols_coord, indexing="ij")

        pix_rows.append(pd.DataFrame({
            "chrom1": info_i[0],
            "start1": rr.ravel(),
            "end1": rr.ravel() + res,
            "chrom2": info_j[0],
            "start2": cc.ravel(),
            "end2": cc.ravel() + res,
            "region_id": key_region_id,
            "name": name
        }))

    all_pix_from_boxes = pd.concat([all_pix_from_boxes] + pix_rows, ignore_index=True)
    all_boxes_df = pd.concat([all_boxes_df, pd.DataFrame(rows)], ignore_index=True)

    all_pixels_df = all_pixels_df.merge(
        all_boxes_df[["region_id", "intensity"]],
        on="region_id",
        how="right"
        )

    return all_pixels_df, all_boxes_df, all_pix_from_boxes


def filter_down_diag(pix_df, box_df, pix_in_box_df):

    filtered_box = box_df[(box_df["start1"] >= box_df["start2"]) | (box_df["end1"] >= box_df["end2"])]
    discard_region_ids = set(filtered_box["region_id"])

    filtered_box_df = box_df.drop(filtered_box.index).reset_index(drop=True)
    filtered_pix_df = pix_df[~pix_df["region_id"].isin(discard_region_ids)].copy().reset_index(drop=True)
    filtered_pix_in_box_df = pix_in_box_df[~pix_in_box_df["region_id"].isin(discard_region_ids)].copy().reset_index(drop=True)

    return filtered_pix_df, filtered_box_df, filtered_pix_in_box_df


def scaled_zf(raw, seg, ref, chunk_size, z_min):

    H, W = raw.shape
    candi = np.zeros((H, W))

    seg_labeled = skimage_label(seg, connectivity=1)
    seg_regions = regionprops(seg_labeled, intensity_image=ref)

    for r in seg_regions:
        minr, minc, maxr, maxc = r.bbox
        box = raw[minr:maxr, minc:maxc].ravel()


        up = max(0, minr - (maxr - minr))
        up2 = max(0, minr - 2 * (maxr - minr))

        down = min(chunk_size, maxr + (maxr - minr))
        down2 = min(chunk_size, maxr + 2 * (maxr - minr))
        
        left = max(0, minc - (maxc - minc))
        left2 = max(0, minc - 2 * (maxc - minc))

        right = min(chunk_size, maxc + (maxc - minc))
        right2 = min(chunk_size, maxc + 2 * (maxc - minc))
            
        sig = 0

        if down != down2 and up != up2 and left != left2 and right != right2:

            fetch_up = raw[up:minr, minc:maxc].ravel()
            fetch_up2 = raw[up2:up, minc:maxc].ravel()
            fetch_down = raw[maxr:down, minc:maxc].ravel()
            fetch_down2 = raw[down:down2, minc:maxc].ravel()            
            fetch_left = raw[minr:maxr, left:minc].ravel()
            fetch_left2 = raw[minr:maxr, left2:left].ravel()            
            fetch_right = raw[minr:maxr, maxc:right].ravel()
            fetch_right2 = raw[minr:maxr, right:right2].ravel()


            mean_obj = np.mean(box)

            context = np.concatenate([fetch_up, fetch_down, fetch_left, fetch_right])
            context2 = np.concatenate([fetch_up2, fetch_down2, fetch_left2, fetch_right2])
        
            mean_bgd = np.mean(context) if context.size else 0
            std_bgd = np.std(context) if context.size else 0
            mean_bgd2 = np.mean(context2) if context2.size else 0
            std_bgd2 = np.std(context2) if context2.size else 0

            if std_bgd != 0 and std_bgd2 != 0 :
                sig1 = (mean_obj - mean_bgd) / std_bgd
                sig2 = (mean_obj - mean_bgd2) / std_bgd2

                if max(sig1, sig2) >= z_min or r.mean_intensity >= 0.06:
                    candi[minr:maxr, minc:maxc] = 1  

        else:
            
            fetch_up = raw[up:minr, minc:maxc].ravel()
            fetch_down = raw[maxr:down, minc:maxc].ravel()
            fetch_left = raw[minr:maxr, left:minc].ravel()
            fetch_right = raw[minr:maxr, maxc:right].ravel()
            
            mean_obj = np.mean(box)

            context = np.concatenate([fetch_up, fetch_down, fetch_left, fetch_right])

            mean_bgd = np.mean(context) if context.size else 0
            std_bgd = np.std(context) if context.size else 0

            if std_bgd != 0:
                sig = (mean_obj - mean_bgd) / std_bgd

                if sig >= z_min or r.mean_intensity >= 0.06:
                    candi[minr:maxr, minc:maxc] = 1    

    return candi