import os
import sys
from typing import Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes
import datatools.matplotsoccer as mps

class TraceSnapshot:
    def __init__(self, traces: pd.DataFrame, visible_area: Tuple = None, play_left_to_right: bool = True, field_dims = (105, 68)):
        self.traces = traces.copy()
        self.visible_area = visible_area
        self.field_dims = field_dims

        if play_left_to_right:
            x_cols = [c for c in self.traces.columns if c.endswith("_x")] # Away team
            y_cols = [c for c in self.traces.columns if c.endswith("_y")] # Away team
            self.traces[x_cols] = field_dims[0] - self.traces[x_cols].values
            self.traces[y_cols] = field_dims[1] - self.traces[y_cols].values

    @staticmethod
    def plot_players(xy: pd.DataFrame, ax: axes.Axes):
        x = xy[xy.columns[0::2]].values[-1]
        y = xy[xy.columns[1::2]].values[-1]

        players = [c[0:3] for c in xy.columns[0::2]]
        color = "tab:red" if players[0][0] == "H" else "tab:blue" # Home team is red, Away team is blue

        ax.scatter(x, y, s=750, c=color, edgecolors=color, linewidths=3, zorder=2) # 시각화
        #ax.scatter(x, y, s=2000, c=color, edgecolors=color, linewidths=3, zorder=2)

        for p in players:
            player_xy = xy[[f"{p}_x", f"{p}_y"]].values
            player_num = int(p[1:])
            ax.plot(player_xy[:, 0], player_xy[:, 1], c=color, ls="--", zorder=0, lw=2.5)

            ax.annotate(
                player_num,
                xy=player_xy[-1],
                ha="center",
                va="center",
                color="w",
                fontsize=18, # 시각화
                #fontsize=30,
                fontweight="bold",
                zorder=3,
            )

    def plot(self, focus_xy: Tuple[float, float] = None, color ="green", save_format=None):
        figsize = (20.8, 14.4) if focus_xy is None else (10, 10)
        fig, ax = plt.subplots(figsize=figsize)
        mps.field(color, fig, ax, show=False)

        if focus_xy is not None:
            ax.set_xlim(focus_xy[0] - 20, focus_xy[0] + 20)
            ax.set_ylim(focus_xy[1] - 20, focus_xy[1] + 20)

        # --- visible_area 그리기 ---
        if self.visible_area is not None:
            # self.visible_area: list of (x,y) tuples, e.g. length-4 polygon
            pts = list(self.visible_area) + [self.visible_area[0]]
            xs, ys = zip(*pts)
            ax.fill(xs, ys,
                    color=(236/256, 236/256, 236/256, 0.5), #red
                    alpha=0.5,
                    edgecolor=(200/256, 200/256, 200/256, 0.7),
                    linewidth=2,
                    zorder=1)
            
        traces = self.traces.dropna(axis=1, how="all")
        xy_cols = [c for c in traces.columns if c.endswith("_x") or c.endswith("_y")]
        team1_xy = traces[[c for c in xy_cols if c[0] == "H"]]
        team2_xy = traces[[c for c in xy_cols if c[0] == "A"]]
        
        if not team1_xy.empty:
            self.plot_players(team1_xy, ax)
        if not team2_xy.empty:
            self.plot_players(team2_xy, ax)

        if "B00_x" not in traces.keys():
            print("not exist ball trajectory")
            # 시각화: 범례 크기
            home_handle = plt.Line2D([], [], color='tab:red', marker='o', linestyle='None', markersize=10, label='Home')
            away_handle = plt.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=10, label='Away')
            ax.legend(handles=[home_handle, away_handle], loc='lower left', fontsize=20)
            # home_handle = plt.Line2D([], [], color='tab:red', marker='o', linestyle='None', markersize=30, label='Home')
            # away_handle = plt.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=30, label='Away')
            # ax.legend(handles=[home_handle, away_handle], loc='lower left', fontsize=30)
        else:
            ball_x = traces["B00_x"].values
            ball_y = traces["B00_y"].values
            ax.scatter(ball_x[-1], ball_y[-1], s=600, c="black", edgecolors="k", marker="o", zorder=4)
            ax.plot(ball_x[-30:], ball_y[-30:], "k", zorder=3)
                

            home_handle = plt.Line2D([], [], color='tab:red', marker='o', linestyle='None', markersize=30, label='Home')
            away_handle = plt.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=30, label='Away')
            ball_handle = plt.Line2D([], [], color='k', marker='o', linestyle='None', markersize=30, label='Ball')
            ax.legend(handles=[home_handle, away_handle, ball_handle], loc='lower left', fontsize=30)

        if save_format == "pdf":
            plt.savefig(f'snapshot.pdf', bbox_inches="tight", dpi=300)
            plt.show()
        elif save_format == "png":
            plt.savefig(f'freeze_frame.png', bbox_inches="tight", dpi=300)
            plt.show()

        return fig, ax