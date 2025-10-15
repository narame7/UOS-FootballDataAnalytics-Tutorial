"""Implements the labels used in each compoment."""
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from tqdm import tqdm
import scipy.signal as signal
import numpy as np
import pandas as pd
import imputer.config as config
import matplotlib.pyplot as plt

# def coordinates(events: pd.DataFrame, traces: pd.DataFrame):
#     """
#     Convert raw data into label DataFrame using coordinates.
#     """

#     tracking_by_period = {
#         period: df.reset_index(drop=True)
#         for period, df in traces.groupby("period_id")
#     }

#     position_cols = [col for col in traces.columns if col.endswith("_x") or col.endswith("_y")]
#     position_idx = []

#     for idx, row in tqdm(events.iterrows()):
#         copy_tracking_data = tracking_by_period[row.period_id]
#         closest_idx = copy_tracking_data["time"].sub(row.time_seconds).abs().idxmin()
#         position_idx.append(closest_idx)

#     # shape: (n_events, 40x2)
#     # ex) [x1, y1, x2, y2, x3, y3, ...]
#     return traces.loc[position_idx, position_cols].reset_index(drop=True)

# def velocity(events: pd.DataFrame, traces: pd.DataFrame):
#     """
#     Convert raw data into label DataFrame using velocity.
#     """

#     tracking_by_period = {
#         period: df.reset_index(drop=True)
#         for period, df in traces.groupby("period_id")
#     }
#     velocity_cols = [col for col in traces.columns if col.endswith("_vx") or col.endswith("_vy")]
#     velocity_idx = []

#     for idx, row in tqdm(events.iterrows()):
#         copy_tracking_data = tracking_by_period[row.period_id]
#         closest_idx = copy_tracking_data["time"].sub(row.time_seconds).abs().idxmin()

#         velocity_idx.append(closest_idx)

#     # shape: (n_events, 40x2)
#     # ex) [vx1, vy1, vx2, vy2, vx3, vy3, ...]
#     return traces.loc[velocity_idx, velocity_cols].reset_index(drop=True)

def coordinates(events: pd.DataFrame, traces: pd.DataFrame):
    """
    Convert raw data into label DataFrame using coordinates.
    """

    tracking_by_period = {
        period: df.reset_index(drop=True)
        for period, df in traces.groupby("period_id")
    }

    position_cols = sorted([col for col in traces.columns if col.endswith("_x") or col.endswith("_y")])
    positions = []

    for period_id, group_events in events.groupby("period_id"):
        copy_tracking_data = tracking_by_period[period_id]

        for row in group_events.itertuples():
            closest_idx = copy_tracking_data["time"].sub(row.time_seconds).abs().idxmin()
            positions.append(copy_tracking_data.loc[closest_idx, position_cols])
        
    # shape: (n_events, 40x2)
    # ex) [x1, y1, x2, y2, x3, y3, ...]
    return pd.DataFrame(positions).reset_index(drop=True)

def velocity(events: pd.DataFrame, traces: pd.DataFrame):
    """
    Convert raw data into label DataFrame using velocity.
    """

    tracking_by_period = {
        period: df.reset_index(drop=True)
        for period, df in traces.groupby("period_id")
    }
    velocity_cols = [col for col in traces.columns if col.endswith("_vx") or col.endswith("_vy")]
    velocity = []

    for period_id, group_events in events.groupby("period_id"): 
        copy_tracking_data = tracking_by_period[period_id]

        for row in group_events.itertuples():
            closest_idx = copy_tracking_data["time"].sub(row.time_seconds).abs().idxmin()
            velocity.append(copy_tracking_data.loc[closest_idx, velocity_cols])

    # shape: (n_events, 40x2)
    # ex) [vx1, vy1, vx2, vy2, vx3, vy3, ...]
    return pd.DataFrame(velocity).reset_index(drop=True)

def identifier(events: pd.DataFrame, traces: pd.DataFrame):
    """
    Convert raw data into label DataFrame using identifier.
    """

    traces = traces.set_index(["period_id", "time"])

    all_identifier = []
    for row in events.itertuples():
        pos = traces.loc[(row.period_id, row.time_seconds)]
        all_identifier.append({f"{col.split('_')[0]}_id": f"{col.split('_')[0]}" for col in pos.columns})

    # shape: (n_events, n_players)
    # ex) [id1, id2, id3, ...]
    return pd.DataFrame(all_identifier)

