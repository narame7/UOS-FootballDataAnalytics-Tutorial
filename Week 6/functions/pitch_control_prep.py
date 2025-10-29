import re

import numpy as np
import pandas as pd
from tqdm import tqdm

import functions.pitch_control as PC
import config

class PitchControlCalculator:
    def __init__(self, merged_df, traces_df, events_df, team_sheets_lookup):
        self.merged_df = merged_df
        self.traces_df = traces_df
        self.events_df = events_df
        self.team_sheets_lookup = team_sheets_lookup

    def frame_flatten_df(self):
        melted = []
        tracking_ids = [col[:-2] for col in self.traces_df.columns if re.fullmatch(r'[HA]\d{2}_x', col)]
        for i, row in enumerate(tqdm(self.traces_df.itertuples(), total=len(self.traces_df))):
            frame_id = row.frame_id
            period_id = row.period_id

            for base_id in tracking_ids:
                x = getattr(row, f"{base_id}_x", np.nan)
                y = getattr(row, f"{base_id}_y", np.nan)
                vx = getattr(row, f"{base_id}_vx", np.nan)
                vy = getattr(row, f"{base_id}_vy", np.nan)

                # if pd.isna(x) or pd.isna(y):
                #     continue

                melted.append({
                    "key": base_id,
                    "period_id": period_id,
                    "time": row.time,
                    "frame_id": frame_id,
                    "team_id": self.team_sheets_lookup[base_id]["team_id"],
                    "team_on_ball": row.ball_owning_team_id== self.team_sheets_lookup[base_id]["team_id"],
                    "position": self.team_sheets_lookup[base_id]['position'],
                    "x": x ,
                    "y": y ,
                    "vx":vx,
                    "vy":vy,
                    "ballx": getattr(row, "B00_x", np.nan),
                    "bally": getattr(row, "B00_y", np.nan)
                })
        return pd.DataFrame(melted)
    
    def _flip_lr(self, side: str) -> str:
        return 'LEFT' if side == 'RIGHT' else 'RIGHT'

    def _home_attacking_side_in_current_period(self, mapping_home: str, period_id: int, flipped: bool) -> str:
        """
        mapping_home: period 1에서 Home의 공격 방향 ('LEFT' or 'RIGHT')
        period_id: 현재 프레임의 period (1 or 2)
        flipped: 좌표를 flip 했는지 여부 (frame_df에 적용된 것과 동일)
        반환: (좌표가 flip된 상태 기준으로) 현재 period에서 Home팀의 공격 방향
        """
        side = mapping_home
        if period_id == 2:
            side = self._flip_lr(side)
        return side  # 'LEFT' or 'RIGHT'

    def _team_on_ball_is_home(self, frame_df) -> bool:
        # team_on_ball == True 인 row의 'key'가 'H..'면 Home, 'A..'면 Away로 가정
        k = str(frame_df.loc[frame_df['team_on_ball'] == True, 'key'].iloc[0])
        return k.startswith('H')

    def _plot_comparison(self, PPCF, team_half: str = None, attacking_area: bool = False, attacking_penalty_area: bool = False) -> tuple:
        """
        PPCF: 2D numpy array
        team_half: 'LEFT' or 'RIGHT' (공격 팀 기준 공격 방향의 절반)
        attacking_area: True면 팀의 '공격하는 절반'만 사용해서 비율 계산
        """
        nrows, ncols = PPCF.shape
        partial_mask = np.zeros_like(PPCF, dtype=bool)

        if attacking_area:
            if team_half == 'RIGHT':
                partial_mask[:, ncols // 2:] = True
            else:  # 'LEFT'
                partial_mask[:, :ncols // 2] = True

        elif attacking_penalty_area:
            if team_half == 'RIGHT':
                partial_mask[14:54, 88:] = True
            else:  # 'LEFT'
                partial_mask[14:54, :27] = True
        else:
            partial_mask = np.ones_like(PPCF, dtype=bool)

        attacking_mask = ((PPCF > 0.5) & partial_mask).astype(int)
        defending_mask = ((PPCF < 0.5) & partial_mask).astype(int)

        total_grids = np.sum(partial_mask)
        attacking_grids = np.sum(attacking_mask)
        defending_grids = np.sum(defending_mask)

        # 0 나눗셈 방지
        if total_grids == 0:
            return 0.0, 0.0

        attacking_ratio = attacking_grids / total_grids
        defending_ratio = defending_grids / total_grids
        return attacking_ratio, defending_ratio

    def _prepare_for_PC(self, melted_df, frame_id):
        # period 1에서의 공격 방향 매핑 (Home 기준)
        self.mapping = (
            self.events_df[self.events_df['period_id'] == 1][['team', 'attack_direction']]
            .drop_duplicates()
            .set_index('team')['attack_direction']
            .to_dict()
        )

        results_df =  melted_df.dropna(subset=['x', 'y'])
        frame_df = results_df[results_df["frame_id"] == frame_id].copy()
        current_period_id = int(frame_df['period_id'].iloc[0])

        # 좌표 flip 여부 (기존 로직 유지)
        flip = (
            (self.mapping['Home'] == 'RIGHT' and current_period_id == 2) or
            (self.mapping['Home'] == 'LEFT'  and current_period_id == 1)
        )
        home_side_now = self._home_attacking_side_in_current_period(self.mapping['Home'], current_period_id, flip)
        
        # 현재 소유 팀이 Home인지 여부에 따라, 실제 공격 팀의 하프 결정
        if self._team_on_ball_is_home(frame_df):
            team_half = home_side_now
        else:
            team_half = self._flip_lr(home_side_now)

        if flip:
            for col in ["x", "ballx"]:
                frame_df[col] = config.field_length - frame_df[col].values
            for col in ["y", "bally"]:
                frame_df[col] = config.field_width - frame_df[col].values
            for col in ["vx", "vy"]:
                frame_df[col] = -frame_df[col].values
            # (좌표가 flip된 상태 기준으로) 현재 period에서 Home의 공격 방향
            
        return frame_df, team_half
    
    def calculate_model_pitch_control(self, melted_df, frame_id, attacking_only=False, attacking_penalty_area_only=False, show_info=False):
        frame_df, team_half = self._prepare_for_PC(melted_df, frame_id)
        params = PC.default_model_params()
        PPCFa, _, _ = PC.generate_pitch_control_for_event(
            frame_df.iloc[0],
            frame_df,
            params,
            field_dimen=(config.field_length, config.field_width),
            n_grid_cells_x=105,
            n_grid_cells_y=68,
            offsides=False,
        )

        attacking_team_abb = frame_df.loc[frame_df['team_on_ball'] == True].iloc[0, 0][0]

        fig, ax = PC.plot_pitchcontrol_for_event(PPCFa, frame_df.iloc[0], frame_df, attacking_team_abb)

        frame_time = float(frame_df.iloc[0]['time'])
        attacking_ratio, defending_ratio = self._plot_comparison(
            PPCFa,
            team_half=team_half,
            attacking_area=attacking_only,  # ← 상대 진영(공격 하프)만 반영
            attacking_penalty_area = attacking_penalty_area_only  # ← 상대 페널티 에어리어만 반영
        )
        if attacking_only:
            info_text = (
                f"time: {frame_time:.2f}\n"
                f"Attacking Half Control: {attacking_ratio:.1%}\n"
                f"Defending Half Control: {defending_ratio:.1%}"
            )
        elif attacking_penalty_area_only:
            info_text = (
                f"time: {frame_time:.2f}\n"
                f"Attacking Control in PA: {attacking_ratio:.1%}\n"
                f"Defending Control in PA: {defending_ratio:.1%}"
            )
        else:
            info_text = (
                f"time: {frame_time:.2f}\n"
                f"Attacking Control: {attacking_ratio:.1%}\n"
                f"Defending Control: {defending_ratio:.1%}"
            )
        if show_info:
            if team_half == 'LEFT':
                ax.text(
                    0.67, 0.85, info_text,
                    transform=ax.transAxes,
                    fontsize=12,
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.5)
                )
            else: 
                ax.text(
                    0.02, 0.85, info_text,
                    transform=ax.transAxes,
                    fontsize=12,
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.5)
                )
        fig.set_size_inches(12, 8)
        return PPCFa, ax, frame_df
