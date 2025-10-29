import numpy as np
import pandas as pd

class GamePreprocessor:
    def __init__(self, traces_df:pd.DataFrame, events_df:pd.DataFrame, teams_df:pd.DataFrame):
        self.traces_df = traces_df
        self.events_df = events_df
        self.teams_df = teams_df

    def _add_direction(self):
        self.teams_df["tracking_id"] = self.teams_df.apply(
            lambda t: f'{t["team"][0]}{t["xID"]:02d}', 
            axis=1
        )
        self.events_df['player_code'] = self.events_df['player_id'].apply(
            lambda x: self.teams_df.loc[self.teams_df["pID"] == x, "tracking_id"].values[0]
            if pd.notna(x) and (x in self.teams_df["pID"].values)
            else None
        )
        self.events_df['team'] = np.where(self.events_df['player_code'].str.startswith('H'), 'Home', 'Away')

        pos_name = self.events_df.get('position_name', pd.Series(dtype=str))
        main_pos = self.events_df.get('main_position', pd.Series(dtype=str))
        position  = self.events_df.get('position', pd.Series(dtype=str))

        is_gk = (
            pos_name.isin(['GK', 'TW']) |
            main_pos.isin(['GK', 'TW']) |
            position.isin(['GK', 'TW'])
        )
        # 2. GK 이벤트만 필터링하는 조건 생성 (반복 방지)
        # is_gk = (
        #     (self.events_df['position_name'].isin(['GK', 'TW'])) |
        #     (self.events_df['main_position'].isin(['GK', 'TW']))
        # )
        # 3. 홈팀 GK의 전반전 평균 위치 '한번만' 계산
        home_gk_x_p1 = self.events_df.loc[
            (self.events_df['team'] == 'Home') & (self.events_df['period_id'] == 1) & is_gk,
            'start_x'
        ].mean()

        # 4. 모든 경우의 수(홈/원정, 전/후반)에 대한 방향 딕셔너리 생성
        if home_gk_x_p1 < 52.5:
            # 홈팀이 전반에 오른쪽으로 공격
            direction_map = {
                ('Home', 1): 'RIGHT', ('Away', 1): 'LEFT',
                ('Home', 2): 'LEFT',  ('Away', 2): 'RIGHT'
            }
        else:
            # 홈팀이 전반에 왼쪽으로 공격
            direction_map = {
                ('Home', 1): 'LEFT',  ('Away', 1): 'RIGHT',
                ('Home', 2): 'RIGHT', ('Away', 2): 'LEFT'
            }

        # 5. 생성한 딕셔너리를 사용해 'attack_direction' 컬럼을 '한번에' 매핑
        # - team과 period_id를 조합하여 key로 사용
        key = pd.MultiIndex.from_frame(self.events_df[['team', 'period_id']])
        self.events_df['attack_direction'] = key.map(direction_map)
    
    def preprocess(self):
        self._add_direction()

        events_for_merge = self.events_df.copy()
        events_for_merge.rename(columns={'time_seconds': 'time'}, inplace=True)
        events_for_merge.loc[events_for_merge['period_id'] == 2, 'time'] += 2700
        
        merged_df = pd.merge_asof(
            left=events_for_merge.sort_values('time'),
            right=self.traces_df.sort_values('time'),
            on='time',
            by='period_id'
        )
        merged_df = merged_df.sort_values(['period_id', 'time']).reset_index(drop=True)
        team_sheets_lookup = {
            row["tracking_id"]: {
                "player_id": row["pID"],
                "team_id":   row["tID"],
                "position":  row["position"],
                "team": row["team"]
            }
            for _, row in self.teams_df.iterrows()
        }
        
        self.merged_df = merged_df
        self.team_sheets_lookup = team_sheets_lookup

        return self # 다른 메서드와 체이닝(chaining)을 위해 self 반환 (선택사항)
