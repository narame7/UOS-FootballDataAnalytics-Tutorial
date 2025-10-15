import os
import numpy as np
import pandas as pd

import scipy.signal as signal
from .utils import load_single_json, load_jsonl
import imputer.config as config

def rename_files(data_path):
    match_id_lst = os.listdir(data_path)
    for match_id in match_id_lst:
        match_path = os.path.join(data_path, match_id)
        if not os.path.isdir(match_path):
            continue
        print(f"\nProcessing directory: '{match_path}'")
        
        meta_path = os.path.join(match_path, f"{match_id}_metadata.json")
        meta_data = load_single_json(meta_path)

        if meta_data:
            # 날짜와 제목 정보 추출
            match_date = meta_data['match_datetime'][:10]  # "YYYY-MM-DD"
            match_title = meta_data['match_title'].replace(' ', '') # 공백 제거
            
            # .info 파일명 생성 및 파일 만들기
            info_filename = f"{match_date}_{match_title}.info"
            info_filepath = os.path.join(match_path, info_filename)
            
            if not os.path.exists(info_filepath):
                open(info_filepath, 'a').close()
                print(f"  ✅ 정보 파일 생성: '{info_filename}'")
            else:
                print(f"  - 정보 파일이 이미 존재합니다.")

        # --- 2. 기존 이벤트 파일 이름 변경 ---
        files_in_match_dir = os.listdir(match_path)
        renamed_count = 0
        for filename in files_in_match_dir:
            cleaned_filename = filename.strip()
            new_filename = None

            if cleaned_filename.endswith("_1st Half.json"):
                new_filename = f"{match_id}_1_event_data.json"
            elif cleaned_filename.endswith("_2nd Half.json"):
                new_filename = f"{match_id}_2_event_data.json"

            if new_filename:
                old_filepath = os.path.join(match_path, filename)
                new_filepath = os.path.join(match_path, new_filename)
                
                if os.path.exists(new_filepath):
                    continue # 이미 변경된 파일은 조용히 넘어감
                
                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"  - 이름 변경: '{filename}' -> '{new_filename}'")
                    renamed_count += 1
                except OSError as e:
                    print(f"  - ❌ 이름 변경 오류: {e}")
        
    print("\nAll operations completed.")
    
class OlderPreprocesssor:
    desired_order = [
        'game_id', 'type_name', 'time_seconds', 'period_id','team', 'player_id', 'outcome', 'qualifier',
         'start_x', 'start_y', 'related_id', 'related_x', 'related_y'
    ]

    def __init__(self, events, tracking_data, game_id):
        self.events = events
        self.tracking_data = tracking_data
        self.game_id = int(game_id)
        self.events["game_id"] = self.game_id
        self.event_mapping = {}

    def make_list(self):
        self.events['event_types'] = self.events['event_types'].astype(str).str.strip().str.split()

    def map_event_type(self, event_type):
        if event_type == 'passSucceeded':
            return 'Pass', 1.0
        elif event_type == 'passReceived':
            return 'Receive', 1.0
        elif event_type == 'passFailed':
            return 'Pass', float('nan')
        elif event_type == 'clearance':
            return 'Clearance', float('nan')
        elif event_type == 'goalKickSucceeded':
            return 'Pass', 1.0
        elif event_type in ['keyPass', 'assist']:
            return 'Pass', 1.0
        elif event_type in ['freeKick', 'cornerKick', 'throwIn']:
            return 'Set-piece', float('nan')
        elif event_type in ['cutoff', 'intercept']:
            return 'Interception', float('nan')
        elif event_type in ['shotOnTarget', 'shot', 'goal']:
            return 'Shot', 1.0 
        elif event_type in ['shotMissed', 'shotBlocked']:
            return 'Shot', float('nan')
        elif event_type == 'duelSucceeded':
            return 'Duel', 1.0
        elif event_type in ['dribbleSucceeded', 'dribbleToSpace', 'controlUnderPressure']:
            return 'Take-On', 1.0
        elif event_type == 'possession':
            return 'Carry', 1.0
        elif event_type == 'dribbleFailed':
            return 'Take-On', float('nan')
        elif event_type == 'foulCommitted':
            return 'Foul', float('nan')
        elif event_type in ['ballMissed', 'goalAgainst', 'goalKickFailed']:
            return 'Mistake', float('nan')
        elif event_type == 'crossSucceeded':
            return 'Cross', 1.0        
        elif event_type == 'crossFailed':
            return 'Cross', float('nan')
        elif event_type in ['saveByPunching', 'saveByCatching']:
            return 'Save', float('nan')
        elif event_type == 'block':
            return 'Block', float('nan')
        elif event_type in ['nan', 'hit']:
            return 'NotaPlay', float('nan')
        else:
            return 'Other', float('nan')

    def apply_mapping(self):
        self.events['event_type_main'] = self.events['event_types'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan
        )
        self.events[['type_name', 'outcome']] = self.events['event_type_main'].apply(
            lambda x: pd.Series(self.map_event_type(x))
        )

    def play_left_to_right(self):
        shots = self.events[self.events['type_name'] == 'Shot']

        # 평균 위치 구하기
        avg_shot_loc = shots.groupby(['session', 'home_away'])[['x']].mean().reset_index()

        # 반전이 필요한 조합 추출
        avg_shot_loc['needs_flip'] = (
            ((avg_shot_loc['home_away'] == 'H') & (avg_shot_loc['x'] <= 52.5)) |
            ((avg_shot_loc['home_away'] == 'A') & (avg_shot_loc['x'] > 52.5))
        )

        # 반전 대상 추출
        flip_keys = avg_shot_loc[avg_shot_loc['needs_flip']][['session', 'home_away']].values.tolist()

        # 원본 데이터 복사
        data_flipped = self.events.copy()

        # 반전 적용
        for sess, side in flip_keys:
            condition = (data_flipped['session'] == sess) & (data_flipped['home_away'] == side)
            data_flipped.loc[condition, 'x'] = config.field_length - data_flipped.loc[condition, 'x'].values
            data_flipped.loc[condition, 'y'] = config.field_width - data_flipped.loc[condition, 'y'].values  # Y축도 원하면

        self.events=data_flipped
    
    def add_related_info(self):
        self.events['related_id'] = pd.Series([np.nan] * len(self.events), dtype="object")
        self.events['related_x'] = np.nan
        self.events['related_y'] = np.nan
        for i in range(len(self.events) - 1):
            curr_row = self.events.iloc[i]
            next_row = self.events.iloc[i + 1]
            if curr_row['type_name'] == 'Pass' and next_row['type_name'] == 'Receive':
                self.events.at[i, 'related_id'] = next_row['player_code']
                self.events.at[i, 'related_x'] = next_row['x']
                self.events.at[i, 'related_y'] = next_row['y']

    def filter_events(self, exclude_list=None):
        if exclude_list is not None:
            self.events = self.events[~self.events["type_name"].isin(exclude_list)].reset_index(drop=True)

    def standardize_columns(self):
        self.events = self.events.rename(columns={
            'time': 'time_seconds',
            'x': 'start_x',
            'y': 'start_y',
            'session': 'period_id',
            'event_types': 'qualifier',
            'player_code': 'player_id',
            'home_away': 'team'
        })

        drop_cols = ['frame', 'phase']
        self.events = self.events.drop(columns=[col for col in drop_cols if col in self.events.columns])

        self.events = self.events[[col for col in self.desired_order if col in self.events.columns]]

    def build_teams(self) -> pd.DataFrame:
        team_dict = {'A': 'Away', 'H': 'Home'}
        player_cols = sorted([col[:3] for col in self.tracking_data.columns if col.endswith("_x") and not col.startswith("ball")])
        team_df = pd.DataFrame(
            {
                "player_id": player_cols,
                "xID": [int(p[1:]) for p in player_cols],
                "team": [team_dict[p[0]] for p in player_cols],
                "game_id": self.game_id,
                "player": player_cols,
                "team_id": [f"{p[0]}{self.game_id}" for p in player_cols]
            }
        )
        # players = self.events[['player_code', 'home_away']].drop_duplicates().reset_index(drop=True).dropna()
        # # players['xID'] = players.groupby('home_away').cumcount()
        # players['xID'] = players["player_code"].apply(lambda x: int(x[1:]))
        # players['team'] = players['home_away'].map({'A': 'Away', 'H': 'Home'})
        # team_df = players.rename(columns={'player_code': 'player_id'})[['player_id', 'xID', 'team']]
        # team_df['team'] = pd.Categorical(team_df['team'], categories=['Home', 'Away'], ordered=True)

        # team_df["game_id"] = self.game_id
        # team_df["player"] = team_df["player_id"]
        # team_df["team_id"] = team_df.apply(lambda x: f'{x["team"][0]}{x["game_id"]}', axis=1)

        # team_df.sort_values(by=['team', 'xID'], inplace=True)
        # team_df.reset_index(drop=True, inplace=True)
        return team_df
    
    def build_events(self, exclude_list=None):
        self.make_list()
        self.apply_mapping()
        self.play_left_to_right()        
        self.add_related_info()
        self.standardize_columns()

        self.filter_events(exclude_list=exclude_list)#["Receive"])

        self.events["player_id"] = self.events["player_id"].map(self.event_mapping)
        self.events["related_id"] = self.events["related_id"].map(self.event_mapping)
        self.events['team'] = self.events['team'].map({'A': 'Away', 'H': 'Home'})
        self.events["team_id"] = self.events.apply(lambda x: f'{x["team"][0]}{x["game_id"]}', axis=1)
        self.events["event_id"] = range(len(self.events))

        return self.events

    def build_tracking_data(self):
        self.tracking_data = self.tracking_data.rename(columns={
            "session": "period_id",
        })

        home_cols = [col for col in self.tracking_data.columns if col.startswith("H")]
        away_cols = [col for col in self.tracking_data.columns if col.startswith("A")]

        for c in ["x", "y", "vx", "vy", "speed", "accel"]:
            home_col = [col for col in home_cols if col.endswith(f"_{c}")]
            away_col = [col for col in away_cols if col.endswith(f"_{c}")]

            home_mapping = {col: col.replace(col[1:3], f"{i:02d}") for i, col in enumerate(sorted(home_col))}
            away_mapping = {col: col.replace(col[1:3], f"{i:02d}") for i, col in enumerate(sorted(away_col))}

            self.tracking_data = self.tracking_data.rename(columns={**home_mapping, **away_mapping})

        for prev_id, new_id in home_mapping.items():
            self.event_mapping[f"{prev_id[:3]}"] = f"{new_id[:3]}"
        for prev_id, new_id in away_mapping.items():
            self.event_mapping[f"{prev_id[:3]}"] = f"{new_id[:3]}"

        x_cols = [col for col in self.tracking_data.columns if col.endswith("_x")]
        y_cols = [col for col in self.tracking_data.columns if col.endswith("_y")]

        self.tracking_data[x_cols] = self.tracking_data[x_cols] / 108 * config.field_length
        self.tracking_data[y_cols] = self.tracking_data[y_cols] / 72 * config.field_width

        self.tracking_data[x_cols] = np.clip(self.tracking_data[x_cols], 0, config.field_length)
        self.tracking_data[y_cols] = np.clip(self.tracking_data[y_cols], 0, config.field_width)

        home_x_cols = [col for col in x_cols if col.startswith("H")]
        away_x_cols = [col for col in x_cols if col.startswith("A")]
        home_y_cols = [col for col in y_cols if col.startswith("H")]
        away_y_cols = [col for col in y_cols if col.startswith("A")]

        for period_id in self.tracking_data["period_id"].unique():
            home_x = np.nanmean(self.tracking_data.loc[self.tracking_data["period_id"] == period_id, home_x_cols].values[0])
            away_x = np.nanmean(self.tracking_data.loc[self.tracking_data["period_id"] == period_id, away_x_cols].values[0])

            # 원정팀이 왼쪽에서 플레이한다면, 좌우 반전 -> Home이 항상 왼쪽->오른쪽으로 공격
            if away_x < home_x:
                self.tracking_data.loc[self.tracking_data["period_id"] == period_id, home_x_cols] = config.field_length - self.tracking_data.loc[self.tracking_data["period_id"] == period_id, home_x_cols].values
                self.tracking_data.loc[self.tracking_data["period_id"] == period_id, away_x_cols] = config.field_length - self.tracking_data.loc[self.tracking_data["period_id"] == period_id, away_x_cols].values
                self.tracking_data.loc[self.tracking_data["period_id"] == period_id, home_y_cols] = config.field_width - self.tracking_data.loc[self.tracking_data["period_id"] == period_id, home_y_cols].values
                self.tracking_data.loc[self.tracking_data["period_id"] == period_id, away_y_cols] = config.field_width - self.tracking_data.loc[self.tracking_data["period_id"] == period_id, away_y_cols].values

        ball_cols = [col for col in self.tracking_data.columns if col.startswith("ball")]
        self.tracking_data = self.tracking_data.drop(columns=ball_cols)

        return self.tracking_data

class NewPreprocessor:
    desired_order = [
        'type_name', 'time_seconds', 'team_name', 'team_id',
        'player_id', 'outcome', 'event_id',
        'timestamp', 'qualifier', 'period_id', 'team', 'start_x', 'start_y', 
        'position', 'related_x', 'related_y', 'related_id','attack_direction',
    ]
    def __init__(self, game_id: int, data_path: str):
        self.game_id = game_id
        self.game_path = os.path.join(data_path, game_id)

        # Load metadata
        self.meta_data = load_single_json(file_path=os.path.join(self.game_path, f"{game_id}_metadata.json"))

        # 경기 별 경기장 크기가 다름.
        self.ground_width = self.meta_data['ground_width']
        self.ground_height = self.meta_data['ground_height']
        self.fps = self.meta_data['fps']

        # Load team info
        self.teams = self._build_teams(self.meta_data['home_team'], self.meta_data['away_team'])

        # Load event data
        self.events = self._load_event_data()
        self.events["event_id"] = range(len(self.events))

        self.traces = self._load_traces()

    @staticmethod
    def _build_teams(home_team_info, away_team_info):
        def build_team(info, team_name):
            rows = []
            for idx, player in enumerate(info['players']):
                rows.append({
                    'player': player['full_name_en'],
                    'position': player['initial_position_name'],
                    'team': team_name,
                    'jID': player['shirt_number'],
                    'pID': player['player_id'],
                    'tID': info['team_id'],
                    'xID': idx
                })
            return pd.DataFrame(rows)

        home_df = build_team(home_team_info, 'Home')
        away_df = build_team(away_team_info, 'Away')

        return pd.concat([home_df, away_df], ignore_index=True)

    def _load_event_data(self):
        # Find the event folder and normalize name if needed
        first_event_path = os.path.join(self.game_path, f"{self.game_id}_1_event_data.json")
        second_event_path = os.path.join(self.game_path, f"{self.game_id}_2_event_data.json")

        first_data = load_single_json(first_event_path)['data']
        second_data = load_single_json(second_event_path)['data']

        return pd.concat([
            pd.DataFrame(first_data),
            pd.DataFrame(second_data)
        ], ignore_index=True).reset_index(drop=True)
    
    def _load_traces(self):
        team_to_tid = {
            'Home': self.teams[self.teams['team'] == 'Home']['tID'].iloc[0],
            'Away': self.teams[self.teams['team'] == 'Away']['tID'].iloc[0]
        }

        first_half_tracking_data = load_jsonl(os.path.join(self.game_path, f"{self.game_id}_1_frame_data.jsonl"))
        second_half_tracking_data = load_jsonl(os.path.join(self.game_path, f"{self.game_id}_2_frame_data.jsonl"))

        all_object_rows = []

        for half_tracking_data in [first_half_tracking_data, second_half_tracking_data]:
            for frame_data in half_tracking_data:
                # Check ball state
                ball_state = frame_data.get('ball_state')
                if ball_state is None or ball_state == 'out':
                    new_ball_state = 'dead'
                    ball_owning_team_id = None
                else:
                    new_ball_state = 'alive'
                    if ball_state == 'home':
                        ball_owning_team_id = team_to_tid['Home']
                    elif ball_state == 'away':
                        ball_owning_team_id = team_to_tid['Away']
                    else:
                        ball_owning_team_id = ball_state

                # 2. Extract current frames base information.
                frame_info = {
                    'game_id': self.game_id,
                    'period_id': frame_data.get('period_order') + 1,
                    'match_time': frame_data.get('match_time'),
                    'frame_id': frame_data.get('frame_index'),
                    'ball_state': new_ball_state,
                    'ball_owning_team_id': ball_owning_team_id,
                }

                for object in ['players', 'balls']:
                    object_list = frame_data.get(object, [])
                    if object_list:
                        for object_data in object_list:
                            row_data = frame_info.copy()
                            row_data.update(object_data)
                            if object == 'balls':
                                row_data['id'] = 'ball'
                            else:
                                row_data['id'] = row_data['player_id']
                            row_data.pop('object')
                            row_data.pop('player_id')
                            all_object_rows.append(row_data)

        tracking_df = pd.DataFrame(all_object_rows)

        return tracking_df

    def adjust_time_and_period(self):
        # 전반 -> 후반으로 이어질 때 시간이 0초부터 시작하지 않는 작업: 사용X(전후반이 이어지는 것을 표현하기 위함)
        #self.events.loc[self.events['period_order'] == 1, 'event_time'] -= 2700000
        self.events['event_time'] = self.events['event_time'].astype(float) * 0.001

        self.events['period_order'] += 1 # 전반: 0 -> 1, 후반: 1 -> 2

        # 후반 시작 시점에 전반 마지막 시점 + 1s으로 조정
        # 이유: 전반과 후반이 이어지는 것을 표현하기 위함. period정보가 없기 때문에 후반에 0초 부터 시작하면, 전반과 후반을 구분할 수 없음
        # first_last_time = self.events[self.events["period_order"] == 1].event_time.max()
        # seconds_start_event = self.events[self.events["period_order"] == 2].iloc[0]
        # seconds_start_time = seconds_start_event.event_time
        # additional_time = (seconds_start_event.event_time - seconds_start_event.period_start_time * 0.001) # 후반전 첫번째 이벤트의 시작 시간
        # self.events.loc[self.events["period_order"] == 2, "event_time"] = self.events.loc[self.events["period_order"] == 2, "event_time"].values - seconds_start_time + first_last_time + additional_time # 0.033초로 설정한 이유: 0.033초는 DFL에서 제공하는 frame rate(30fps)와 일치하기 때문

    def _parse_actiontype(self):
        '''
        여러가지 이벤트 데이터의 액션 타입을 통일하기 위한 작업
        ex) Passes, Key Passes -> Pass
        '''

        # DFL에서 사용하는 actiontype과 매핑
        actiontype_mapping = {
            'Set Piece Defence': 'Clearance',
            'Turnover': 'Mistake',
            'Shots & Goals': 'Shot',
            'Step-in': 'Carry',
            'Aerial Control': 'Aerial Control',
            'Passes' : 'Pass',
            'Tackles' : 'Tackle',
            'Blocks' : 'Block',
            'Set Pieces' : 'Set-piece',
            'Clearances' : 'Clearance',
            'Recoveries' : 'Recovery',
            'Mistakes' : 'Mistake',
            'Duels' : 'Duel',
            'Fouls' : 'Foul',
            'Crosses' : 'Cross',
            'Interceptions' : 'Interception',
            'Offsides' : 'Offside',
            "Passes Received": "Receive",
            "Crosses Received": "Receive",
            "Goals Conceded": "Save",
            "Take-on": "Take-On",
            "Saves": "Save",
            "Defensive Line Supports": "Defensive Line Supports",
            "Own Goals": "Own Goal",
            "Assists": "Other",
            'Key Passes': 'Other', # Key Passes는 Pass or Set-piece랑 같이 나오기 때문에 Pass로 통합할 필요X
        }

        # 우선순위 매핑
        priority_map = {
            "Throw-In": 0,
            "Pass_Corner": 0,
            "Shot_Corner": 0,
            "Pass_Freekick": 0,
            "Shot_Freekick": 0,
            "Goal Kick": 0,
            "Penalty Kicks": 0,

            "Save": 0,
            "Own Goal": 0,
            
            "Aerial Control": 1,
            "Defensive Line Supports": 1,

            "Duel": 2,

            "Tackle": 3,

            "Interception": 4,
            "Block": 4,

            "Receive": 5,
            "Recovery": 5,

            "Mistake": 6,
            "Carry": 6,
            "Take-On": 6,
            "Pass": 6,  
            "Shot": 6,
            "Clearance": 6,
            "Cross": 6,
            "Offside": 6,

            "Foul": 7,
            "Foul Won": 7, # Pass Received + Foul -> 받은 이후에 파울이 발생한 경우
        }
        self.events = self.events.copy()

        # 액션 타입을 DFL에서 사용하는 액션 타입으로 매핑 & Other(Assists, Key Passes)은 제거
        self.events = self.events[self.events["events"].apply(lambda x: any("event_name" in r for r in x))].reset_index(drop=True)
        self.events['events'] = self.events['events'].apply(
            lambda x: [{**r, "event_name": actiontype_mapping[r["event_name"]]} for r in x]
        )
        self.events["events"] = self.events["events"].apply(
            lambda x: [r for r in x if r["event_name"] != "Other"]
        )

        # 1. set_piece simplify: set_piece과 동시에 발생한 이벤트 정보는 set_piece의 부가적인 특성(pass + mistake)이므로 통합
        # ex) pass + set_piece -> set_piece, pass + mistake + set_piece -> set_piece
        not_set_piece = self.events[self.events['events'].apply(lambda x: not any(r['event_name'] == 'Set-piece' for r in x))].copy()
        set_piece = self.events[self.events['events'].apply(lambda x: any(r['event_name'] == 'Set-piece' for r in x))].copy()

        throw_in_cond = set_piece['events'].apply(lambda x: any(r['property'].get('Type') == 'Throw-Ins' for r in x if 'property' in r))
        corner_cond = set_piece['events'].apply(lambda x: any(r['property'].get('Type') == 'Corners' for r in x if 'property' in r))
        freekick_cond = set_piece['events'].apply(lambda x: any(r['property'].get('Type') == 'Freekicks' for r in x if 'property' in r))
        goalkick_cond = set_piece['events'].apply(lambda x: any(r['property'].get('Type') == 'Goal Kicks' for r in x if 'property' in r))
        penalty_kick_cond = set_piece['events'].apply(lambda x: any(r['property'].get('Type') == 'Penalty Kicks' for r in x if 'property' in r))

        pass_cond = set_piece['events'].apply(lambda x: any(r['event_name'] in ['Pass', 'Cross'] for r in x))
        shot_cond = set_piece['events'].apply(lambda x: any(r['event_name'] == 'Shot' for r in x))

        # 새로운 events 컬럼 생성
        set_piece.loc[throw_in_cond, 'events'] = set_piece.loc[throw_in_cond, "events"].apply(lambda x: [{"property": r['property'], "event_name": "Throw-In"} for r in x if r['event_name'] in ['Pass', 'Cross']])
        set_piece.loc[corner_cond & pass_cond, 'events'] = set_piece.loc[corner_cond & pass_cond, "events"].apply(lambda x: [{"property": r['property'], "event_name": "Pass_Corner"} for r in x if r['event_name'] in ['Pass', 'Cross']])
        set_piece.loc[corner_cond & shot_cond, 'events'] = set_piece.loc[corner_cond & shot_cond, "events"].apply(lambda x: [{"property": r['property'], "event_name": "Shot_Corner"} for r in x if r['event_name'] == 'Shot'])
        set_piece.loc[freekick_cond & pass_cond, 'events'] = set_piece.loc[freekick_cond & pass_cond, "events"].apply(lambda x: [{"property": r['property'], "event_name": "Pass_Freekick"} for r in x if r['event_name'] in ['Pass', 'Cross']])
        set_piece.loc[freekick_cond & shot_cond, 'events'] = set_piece.loc[freekick_cond & shot_cond, "events"].apply(lambda x: [{"property": r['property'], "event_name": "Shot_Freekick"} for r in x if r['event_name'] == 'Shot'])
        set_piece.loc[goalkick_cond, 'events'] = set_piece.loc[goalkick_cond, "events"].apply(lambda x: [{"property": r['property'], "event_name": "Goal Kick"} for r in x if r['event_name'] in ['Pass', 'Cross']])
        set_piece.loc[penalty_kick_cond, 'events'] = set_piece.loc[penalty_kick_cond, "events"].apply(lambda x: [{"property": r['property'], "event_name": "Penalty Kicks"} for r in x if r['event_name'] == 'Shot'])
        

        self.events = pd.concat([not_set_piece, set_piece], ignore_index=True).sort_values(by="event_id", kind="mergesort").reset_index(drop=True)

        # 2. mistake 제거: mistake과 동시에 발생한 이벤트 정보는 그 외 이벤트(pass + mistake)의 부가적인 특성이 mistake이므로 mistake 제거
        # ex) pass + mistake -> pass
        not_mistake = self.events[self.events['events'].apply(lambda x: not any(r['event_name'] == 'Mistake' for r in x))].copy()
        mistake = self.events[self.events['events'].apply(lambda x: any(r['event_name'] == 'Mistake' for r in x))].copy()
        mistake['events'] = mistake['events'].apply(
            lambda x: x if len(x) == 1 else [r for r in x if r['event_name'] != 'Mistake']
        )
        self.events = pd.concat([not_mistake, mistake], ignore_index=True).sort_values(by="event_id", kind="mergesort").reset_index(drop=True)

        # 3. cross simplify: cross과 동시에 발생한 Pass 이벤트는 cross의 부가적인 특성(pass + cross)이므로 통합
        # ex) pass + cross -> cross
        non_cross = self.events[self.events['events'].apply(lambda x: not any(r['event_name'] == 'Cross' for r in x))].copy()
        cross = self.events[self.events['events'].apply(lambda x: any(r['event_name'] == 'Cross' for r in x))].copy()
        cross['events'] = cross['events'].apply(
            lambda x: [r for r in x if r['event_name'] != 'Pass']
        )
        self.events = pd.concat([non_cross, cross], ignore_index=True).sort_values(by="event_id", kind="mergesort").reset_index(drop=True)

        # 4. foul & foul won: foul의 부가적인 특성(foul + foul won)을 기준으로 분리
        # foul won: Fouls Won, Penalty Kick Won
        non_foul = self.events[self.events['events'].apply(lambda x: not any(r['event_name'] == 'Foul' for r in x))].copy()
        foul = self.events[self.events['events'].apply(lambda x: any(r['event_name'] == 'Foul' for r in x))].copy()
        foulwon_cond = foul['events'].apply(
            lambda x: any(
                (r['event_name'] == 'Foul') and
                ((r['property'].get('Type', None) == "Fouls Won") or
                (r['property'].get('Penalty Kick Won', 'False') == 'True')) # 'False', 'True'가 boolean이 아닌 string으로 되어있음
                for r in x
            )
        )
        foulwon = foul[foulwon_cond].copy()
        foul = foul[~foulwon_cond].copy()
        foulwon['events'] = foulwon['events'].apply(
            lambda x: [{**r, "event_name": "Foul Won"} if r['event_name'] == 'Foul' else r for r in x]
        )
        self.events = pd.concat([non_foul, foul, foulwon], ignore_index=True).sort_values(by="event_id", kind="mergesort").reset_index(drop=True)
        
        new_rows = []
        for _, row in self.events.iterrows():
            event_list = row["events"]
            
            sorted_event_list = sorted(event_list, key=lambda e: priority_map.get(e["event_name"]))

            # Debugging: 동일한 우선순위를 가진 이벤트가 있으면 오류
            priorities = [priority_map.get(e["event_name"]) for e in sorted_event_list]
            if len(priorities) != len(set(priorities)):
                print(row.event_id, event_list)
                raise ValueError(f"Duplicate priority values found for events: {event_list}")
            
            new_rows.extend([
                {**row.to_dict(), "type_name": event["event_name"]}
                for event in sorted_event_list
            ])

        self.events = pd.DataFrame(new_rows)

    def _parse_outcome(self):
        def extract(event_list):
            for event in event_list:
                if 'property' in event and 'Outcome' in event['property']:
                    return 1.0 if event['property']['Outcome'] == 'Succeeded' else float('nan')
            return float('nan')
        self.events['outcome'] = self.events['events'].apply(extract)

    def add_player_info(self):
        name_to_position = {row.player: row.position for row in self.teams.itertuples()}
        name_to_pid = {row.player: row.pID for row in self.teams.itertuples()}

        self.events['position'] = self.events['player_name'].map(name_to_position)
        self.events['player_id'] = self.events['player_name'].map(name_to_pid)

    def play_left_to_right(self):
        '''
        Home팀이 왼쪽에서 오른쪽으로 공격하는 방향으로 변환
        Away팀이 오른쪽에서 왼쪽으로 공격하는 방향으로 변환
        attack_direction: 공격하는 방향. if LEFT, Home팀이 오른쪽에서 왼쪽으로 공격
        '''
        home_team = self.meta_data['home_team']['team_name_en']
        is_home = self.events['team_name'] == home_team
        self.events['team'] = is_home.map({True: 'Home', False: 'Away'})
        flip_xy = (
            (is_home & (self.events['attack_direction'] == 'LEFT')) | # Home팀이 왼쪽으로 공격하고 있으면 flip
            (~is_home & (self.events['attack_direction'] == 'RIGHT')) # Home팀이 오른쪽으로 공격하고 있으면 flip
        )
        self.events.loc[flip_xy, ['x', 'to_x']] = 1 - self.events.loc[flip_xy, ['x', 'to_x']].values
        self.events.loc[flip_xy, ['y', 'to_y']] = 1 - self.events.loc[flip_xy, ['y', 'to_y']].values

        self.events[['x', 'to_x']] *= config.field_length
        self.events[['y', 'to_y']] *= config.field_width

    def add_related_info(self):
        self.events[['related_id', 'related_x', 'related_y']] = np.nan
        for i in range(len(self.events) - 1):
            cur = self.events.iloc[i]
            next = self.events.iloc[i+1:i+10] # 최소 10개 내 receive 이벤트를 찾음

            # 성공한 패스의 경우 receive 이벤트를 찾아서 related_id에 추가
            if cur['type_name'] in ['Pass', 'Cross'] and cur['outcome'] == 1.0:
                receival = next[next['type_name'].isin(['Receive'])]
                if not receival.empty:
                    # game_di = 126298, event_id = 2563: 정호연 선수가 패스한 공을 정호연 선수가 받은 경우 
                    # 결론: passer = receiver인 경우는 사용X. event_xy != observe_xy이기 때문에 feature(observe_xy)로 잘못 들어갈 수 있음
                    if receival.iloc[0]["player_id"] == cur["player_id"]:
                        print(f"passer = receiver -> game_id: {self.game_id}, event_id: {cur['event_id']}, passer: {cur['player_name']}, receiver: {receival.iloc[0]['player_name']}")
                    else:
                        self.events.at[i, 'related_id'] = receival.iloc[0]['player_id']
                        self.events.at[i, 'related_x'] = cur['to_x']
                        self.events.at[i, 'related_y'] = cur['to_y']
                else:
                    print(f"No receival event found for pass: {self.game_id}, event_id: {cur['event_id']}, passer: {cur['player_name']}, qualifier: {cur['events']}")

    def standardize_columns(self):
        self.events = self.events.rename(columns={
            # 'to_x': 'related_x',
            # 'to_y': 'related_y',
            'event_time': 'time_seconds',
            'x': 'start_x',
            'y': 'start_y',
            'period_order': 'period_id',
            'events': 'qualifier'
        })

        self.events = self.events.drop(columns=[
            'period_type', 'period_name', 'period_duration',
            'period_start_time', 'player_shirt_number'
        ])

        # 진짜 컬럼 정렬 (원하면)
        self.events = self.events[[col for col in self.desired_order if col in self.events.columns]]

    def filter_events(self, exclude_list=None):
        if exclude_list is not None:
            self.events = self.events[~self.events["type_name"].isin(exclude_list)].reset_index(drop=True)

    def reorder_columns(self):
        self.events = self.events.drop(columns=[
            'period_type', 'period_name', 'period_duration', 'period_start_time', 'player_shirt_number'
        ])

    def build_events(self, exclude_list=None, left_to_right=True):
        """
        추후 이벤트 필터링/정제 등을 여기에 추가
        exlude_list: ['Pass', 'Shot', 'Foul']: 학습에 사용하지 않는 데이터셋
        left_to_right: True -> Home팀이 왼쪽에서 오른쪽으로 공격하는 방향으로 변환
        """

        # time, period를 DFL에 맞게 변환
        self.adjust_time_and_period()

        # qualifer를 통해 actiontype, outcome을 추출
        self._parse_outcome()     
        self._parse_actiontype()
        
        self.add_player_info()

        if left_to_right:
            self.play_left_to_right()
           
        self.add_related_info()              
        self.standardize_columns()
        self.filter_events(exclude_list=[])

        self.events["event_id"] = range(len(self.events))
        return self.events
    
    def build_traces(self, left_to_right=True):
        """
        Build the tracking data by processing the raw tracking data and adding relevant information.
        left_to_right: True -> Home팀이 왼쪽에서 오른쪽으로 공격하는 방향으로 변환
        """
        team_lookup = {
            pid: f"{team[0]}{xid:02d}"
            for pid, team, xid in self.teams[["pID", "team", "xID"]].values
        }
        team_lookup["ball"] = "Ball"

        # Convert match_time to seconds and adjust for periods
        # self.traces["time_seconds"] = (
        #     self.traces["match_time"] * 0.001
        #     - ((self.traces.period_id > 1) * 45 * 60)
        #     - ((self.traces.period_id > 2) * 45 * 60)
        #     - ((self.traces.period_id > 3) * 15 * 60)
        #     - ((self.traces.period_id > 4) * 15 * 60)
        # )
        self.traces["time"] = self.traces["match_time"] * 0.001

        # Pre-group data by frame_id for faster access: 전/후반이 이어지는 프레임은 동일한 frame_id를 가지기 때문에 주의해야 함.
        # ex) 1st half: 0~2700, 2nd half: 2700~5400
        grouped = self.traces.groupby(["period_id", "frame_id"])
        
        # Initialize the result list
        tracking_df = []
        
        # Iterate over each frame_id group
        for (period_id, frame_id), frame_df in grouped:
            # Dictionary to store combined data for the current frame
            frame_data = {"frame_id": frame_id, "time": frame_df["time"].iloc[0],
                        "period_id": period_id, "ball_state": frame_df["ball_state"].iloc[0],
                        "ball_owning_team_id": frame_df["ball_owning_team_id"].iloc[0]}
            
            # Iterate over rows in the frame_df
            for row in frame_df.itertuples():
                xID = team_lookup.get(row.id, f"unknown_{row.id}")  # Handle missing IDs gracefully
                
                # Clip x and y coordinates to the specified ranges
                # config.field_length = x축 = self.ground_width = 105
                # config.field_width = y축 = self.ground_height = 68
                normalized_x = row.x / self.ground_width * config.field_length
                normalized_y = row.y / self.ground_height * config.field_width
                clipped_x = max(0, min(config.field_length, normalized_x))
                clipped_y = max(0, min(config.field_width, normalized_y))
                
                frame_data[f"{xID}_x"] = clipped_x
                frame_data[f"{xID}_y"] = clipped_y
                frame_data[f"{xID}_speed"] = row.speed
            
            # Append the combined data for the frame
            tracking_df.append(frame_data)
        
        # Convert the list of dictionaries into a DataFrame
        self.traces = pd.DataFrame(tracking_df)

        # 경기에 뛰지 않은 선수들의 정보(컬럼)도 추가 -> 데이터셋 호환(통합)
        all_player_cols = self.teams.apply(lambda x: f'{x["team"][0]}{x["xID"]:02d}', axis=1).values.tolist()
        traces_cols = [col[:3] for col in self.traces.columns if col.endswith("_x") and not col.startswith("B")]
        not_trace_cols = [col for col in all_player_cols if col not in traces_cols]
        not_trace_xy = [[f'{p}_{suffix}' for p in not_trace_cols for suffix in ["x", "y"]]]
        self.traces [not_trace_xy] = np.nan
        
        # 기존 bepro데이터는 전반·후반을 별도로 관리하다 보니 프레임 번호가 중복되는 현상이 발생한다.
        # 후반전의 시작 프레임은 전반전의 마지막 프레임 + 1이 아닌 2,700,000(45분, 2,700초)부터 시작하므로 synchronize가 맞질 않는다.
        # start_frame = self.traces["frame_id"].min()
        # self.traces["frame_id"] = range(start_frame, start_frame + len(self.traces))
        #self.traces["time"] = self.traces["frame_id"] / 30 # fps = 30

        self.traces = self.traces.rename(columns={"Ball_x": "B00_x", "Ball_y": "B00_y", "Ball_speed": "B00_speed"})

        # calculate player velocities: 내부로직에서 전/후반 별도로 분리해서 계산함
 
        if left_to_right:
            x_cols = [col for col in self.traces.columns if col.endswith("_x")]
            y_cols = [col for col in self.traces.columns if col.endswith("_y")]
            home_x_cols = [col for col in x_cols if col.startswith("H")]
            away_x_cols = [col for col in x_cols if col.startswith("A")]

            for period_id in self.traces["period_id"].unique():
                # 전/후반 첫번째 프레임의 평균 x좌표를 구함
                home_x = np.nanmean(self.traces.loc[self.traces["period_id"] == period_id, home_x_cols].values[0])
                away_x = np.nanmean(self.traces.loc[self.traces["period_id"] == period_id, away_x_cols].values[0])

                # 원정팀이 왼쪽에서 플레이를 시작 한다면, 좌우 반전 -> Home이 항상 왼쪽->오른쪽으로 공격
                # 공도 같이 좌우 반전: 학습에는 사용하지 않지만 downstream task에 사용하기 위함
                if away_x < home_x:
                    self.traces.loc[self.traces["period_id"] == period_id, x_cols] = config.field_length - self.traces.loc[self.traces["period_id"] == period_id, x_cols].values
                    self.traces.loc[self.traces["period_id"] == period_id, y_cols] = config.field_width - self.traces.loc[self.traces["period_id"] == period_id, y_cols].values
        self.traces = self.calc_player_velocities()

        return self.traces
  
    def calc_player_velocities(self, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
        """
        Parameters
        -----------
            positions: the tracking DataFrame for home or away team
            home_df, away_df: Team dataframes
            smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
            filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
            window: smoothing window size in # of frames
            polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
            maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN.
        Returns
        -----------
        positions : the tracking DataFrame with columns for speed in the x & y direction and total speed added
        """

        # Remove any existing velocity columns
        speed_cols = [col for col in self.traces.columns if col.endswith("_speed")]
        self.traces = self.traces.drop(columns=speed_cols)

        # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
        dt = 1 / self.fps # fps=30
        # index of first frame in second half
        second_half_idx = self.traces['period_id'].idxmax(axis=0)
        # estimate velocities for players in team
        player_ids = [p[:3] for p in self.traces.columns if p.endswith("_x") and not p.startswith("B")]
        velocity_data = {}
        for player in player_ids: # cycle through players individually
            # difference player positions in timestep dt to get unsmoothed estimate of velicity
            x = pd.Series([float(p) for p in self.traces[player+"_x"]])
            y = pd.Series([float(p) for p in self.traces[player+"_y"]])
            vx = x.diff() / dt
            vy = y.diff() / dt
            if maxspeed>0:
                # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
                raw_speed = np.sqrt(vx**2 + vy**2)
                vx[raw_speed>maxspeed] = np.nan
                vy[raw_speed>maxspeed] = np.nan
            if smoothing:
                if filter_=='Savitzky-Golay':
                    # calculate first half velocity
                    vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                    vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                    # calculate second half velocity
                    vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                    vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                elif filter_=='moving average':
                    ma_window = np.ones( window ) / window
                    # calculate first half velocity
                    vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' )
                    vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )
                    # calculate second half velocity
                    vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' )
                    vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' )
                else:
                    raise ValueError("Invalid filter type. Must be either 'Savitzky-Golay' or 'moving average'")
            # Store player speed in x, y direction, and total speed in the dictionary
            velocity_data[player + "_vx"] = vx
            velocity_data[player + "_vy"] = vy
            velocity_data[player + "_speed"] = np.sqrt(vx**2 + vy**2)
        velocity_df = pd.DataFrame(velocity_data).round(2)
        
        return pd.concat([self.traces, velocity_df], axis=1)
