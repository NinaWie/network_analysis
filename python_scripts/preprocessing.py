import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr
import os

import pickle

# group purposes
further_agg_dict = {
    "arts": "leisure",
    "church": "leisure",
    "nightlife": "leisure",
    "travel": "leisure",
    "vacation": "leisure",
    "other": "leisure",
    "outdoor_city": "leisure",
    "residential": "leisure",
    "restaurant": "leisure",
    "shop": "shop",
    "doctor": "shop",
    "home": "home",
    "office": "work",
    "school": "work",
    "sport": "leisure",
}


def get_coarse_purpose_category(p):
    fine_cat = get_purpose_category(p)
    return further_agg_dict[fine_cat]


def get_purpose_category(p):
    low = p.lower()
    if low == "office" or "conference" in low or "coworking" in low or "work" in low:
        return "office"
    elif (
        "food" in low
        or "restaurant" in low
        or "pizz" in low
        or "salad" in low
        or "ice cream" in low
        or "bakery" in low
        or "burger" in low
        or "sandwich" in low
        or "caf" in low
        or "diner" in low
        or "snack" in low
        or "steak" in low
        or "pub" in low
        or "tea" in low
        or "noodle" in low
        or "chicken" in low
        or "brewery" in low
        or "breakfast" in low
        or "beer" in low
        or "bbq" in low
    ):
        return "restaurant"
    elif (
        "doctor" in low
        or "hospital" in low
        or "medical" in low
        or "emergency" in low
        or "dental" in low
        or "dentist" in low
    ):
        return "doctor"
    elif (
        "bus" in low
        or "airport" in low
        or "train" in low
        or "taxi" in low
        or "station" in low
        or "metro" in low
        or "travel" in low
        or "ferry" in low
    ):
        return "travel"
    elif (
        "store" in low
        or "shop" in low
        or "bank" in low
        or "deli" in low
        or "mall" in low
        or "arcade" in low
        or "boutique" in low
        or "post" in low
        or "market" in low
        or "dealership" in low
    ):
        return "shop"
    elif "bar" in low or "disco" in low or "club" in low or "nightlife" in low or "speakeasy" in low:
        return "nightlife"
    elif "home" in low:
        return "home"
    elif "residential" in low or "building" in low or "neighborhood" in low:
        return "residential"
    elif (
        "entertain" in low
        or "theater" in low
        or "music" in low
        or "concert" in low
        or "museum" in low
        or "art" in low
        or "temple" in low
        or "historic" in low
    ):
        return "arts"
    elif (
        "golf" in low
        or "tennis" in low
        or "dance" in low
        or "sport" in low
        or "gym" in low
        or "hiking" in low
        or "skating" in low
        or "soccer" in low
        or "basketball" in low
        or "surf" in low
        or "stadium" in low
        or "baseball" in low
        or "yoga" in low
    ):
        return "sport"
    elif "school" in low or "college" in low or "university" in low or "student" in low:
        return "school"
    elif "church" in low or "mosque" in low or "spiritual" in low:
        return "church"
    elif "vacation" in low or "hotel" in low or "beach" in low or "tourist" in low or "bed &" in low:
        return "vacation"
    elif (
        "city" in low
        or "park" in low
        or "plaza" in low
        or "bridge" in low
        or "outdoors" in low
        or "playground" in low
        or "lake" in low
        or "pier" in low
        or "field" in low
        or "harbor" in low
    ):
        return "outdoor_city"
    elif low == "leisure":
        return "leisure"
    else:
        return "other"


def save_attributes(node_feat, out_path="individual_assigment"):
    node_feats = node_feat.drop(columns=["center", "node_id", "started_at"])
    if "extent" in node_feats:
        node_feats = node_feat.drop(columns=["extent"])

    # normalize and rename distance from home attribute
    node_feats = node_feats.rename(columns={"distance": "dist_from_home"})
    node_feats["dist_from_home"] = node_feats["dist_from_home"] / 1000  # transform to km

    if "purpose" in node_feats.columns:
        node_feats["purpose"] = node_feats["purpose"].apply(get_coarse_purpose_category)

    # compute distances
    dist = np.zeros((len(node_feats), len(node_feats)))
    for i, node_i in enumerate(node_feats.index):
        for j, node_j in enumerate(node_feats.index):
            diff_x = node_feats.loc[node_i, "x_normed"] - node_feats.loc[node_j, "x_normed"]
            diff_y = node_feats.loc[node_i, "y_normed"] - node_feats.loc[node_j, "y_normed"]
            dist[i, j] = np.linalg.norm([diff_x, diff_y]) / 1000  # transform to km
    pd.DataFrame(dist).to_csv(
        os.path.join(out_path, f"distances.csv"),
        index=False,
    )
    node_feats.to_csv(os.path.join(out_path, f"project_attr.csv"))
    print("saved attributes", node_feats.shape)


def save_net(adjacency, out_path="individual_assigment", save_num=0):
    adj = adjacency.todense()
    adj_df = pd.DataFrame(adj)
    adj_df.to_csv(
        os.path.join(out_path, f"project_weighted_adj_{save_num}.csv"),
        index=False,
    )

    adj_unweighted = (adj > 0).astype(int)
    adj_df = pd.DataFrame(adj_unweighted)
    adj_df.to_csv(
        os.path.join(out_path, f"project_adj_{save_num}.csv"),
        index=False,
    )


def get_common_locs(part_node_feat_list):
    common_ids = part_node_feat_list[0]
    for i in range(1, len(part_node_feat_list)):
        common_ids = pd.merge(
            common_ids, part_node_feat_list[i], how="inner", left_on="location_id", right_on="location_id"
        )
    return common_ids["location_id"].values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=os.path.join("..", "data", "foursquare"),
        help="Which dataset - default Foursquare, could be gc1, gc2, geolife etc",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=os.path.join("..", "data", f"foursquare_graphs_data.pkl"),
        help="Name under which to save the data",
    )
    parser.add_argument(
        "-t",
        "--time_bins",
        type=int,
        default=3,
        help="Number of time bins to save",
    )

    args = parser.parse_args()
    num_bins = args.time_bins
    in_path = args.input
    path = args.out_dir
    os.makedirs(path, exist_ok=1)

    with open(in_path, "rb") as outfile:
        (user_id_list, adjacency_list, node_feat_list) = pickle.load(outfile)

    print("Loaded graph data", len(user_id_list), len(adjacency_list))

    # get the indices of each user
    if "tist" in in_path or "foursquare" in in_path:
        # tist_toph100_userid --> use part 2
        id_list = [int(e.split("_")[2]) for e in user_id_list]
    else:
        id_list = [int(e.split("_")[1]) for e in user_id_list]
    unique_user_ids = np.unique(id_list)
    index_dict = {}
    for user_id in unique_user_ids:
        index_dict[user_id] = np.where(id_list == user_id)[0]

    nr_valid_users = 0
    for user_id in unique_user_ids:
        time_bins_from_dict = index_dict[user_id][:num_bins]
        print("These time bins are available for the graph:", time_bins_from_dict)
        if len(time_bins_from_dict) < num_bins:
            # skip the ones that have only 2 full timeslots
            continue
        test_ids = [user_id_list[item] for item in time_bins_from_dict]
        part_node_feats = [node_feat_list[item] for item in time_bins_from_dict]
        common_locs = get_common_locs(part_node_feats)

        if len(common_locs) < 10:
            print("Graph too small, skip")
            # Too short
            continue

        # make user-wise  directory
        out_path = os.path.join(path, str(user_id))
        os.makedirs(out_path, exist_ok=1)

        print("user id", user_id, "inds", time_bins_from_dict, "save as", out_path)

        # iterate over time steps
        for i_ind, i in enumerate(time_bins_from_dict):
            if len(node_feat_list[i]) == len(common_locs):
                print("already done")
                continue
            prev_node_feat = node_feat_list[i]
            assert len(prev_node_feat) == len(np.unique(prev_node_feat["location_id"]))
            prev_node_feat = prev_node_feat.reset_index().set_index("location_id")
            restricted_feats = prev_node_feat.loc[common_locs]
            adj = adjacency_list[i]
            restricted_adj = adj[restricted_feats["id"]]
            restricted_adj = restricted_adj[:, restricted_feats["id"]]

            node_feat_list[i] = restricted_feats.reset_index().set_index("id")
            adjacency_list[i] = restricted_adj

            # save attributes only ones --> they are all static over time
            if i_ind == 0:
                save_attributes(node_feat_list[i], out_path=out_path)
            # save graphs
            save_net(adjacency_list[i], out_path=out_path, save_num=i_ind)
            print("saved", i, len(common_locs))
        nr_valid_users += 1
    print("Valid users", nr_valid_users, "out of", len(unique_user_ids))
