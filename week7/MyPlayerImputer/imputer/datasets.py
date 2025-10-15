import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(base_path)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

import re
from typing import Dict, List
import numpy as np
import pandas as pd
import scipy.signal as signal
from torch.utils.data import Dataset
import torch
import ast
from tqdm import tqdm

# features.py, labels.py 내부 함수 임포트 (예시)
from PlayerImputer.imputer import features, labels, transform
from sklearn.preprocessing import StandardScaler, RobustScaler
import PlayerImputer.imputer.config as config
from torch.nn.utils.rnn import pad_sequence
from datatools.preprocess import extract_match_id
from torch.utils.data import DataLoader
from PlayerImputer.datatools.utils import compute_camera_coverage, is_inside
from PlayerImputer.imputer.transform import Transform
from datatools.Bepro_preprocessor import OlderPreprocesssor, NewPreprocessor
import vaep as vaep

class ImputerDataset(Dataset):
    def __init__(
        self,
        game_ids: List,
        data_dir: str,
        model: str,
        xfns: List,
        yfns: List,
        window: int = 1,
        actionfilter: bool = True,
        use_transform: bool = True,
        transform: transform = None,
        play_left_to_right: bool = False,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.xfns = xfns
        self.yfns = yfns
        self.window = window
        self.model = model

        self.use_transform = use_transform
        self.transform=transform
        self.play_left_to_right = play_left_to_right

        self.freeze_frame = None

        # 리스트 초기화
        features_lst = []
        labels_lst = []
        n_away_players_active_lst = []
        actor_valid_index_lst = []
        actor_global_index_lst = [] 
        n_away_players_total_lst = []
        freeze_frame_lst = []
        freeze_frame_mask_lst = []
        active_players_dict_lst = []

        if self.model == 'lstm':
            timestamps_lst = []

        for id in tqdm(game_ids):       
            if "DFL" in self.data_dir:
                self.events = pd.read_csv(os.path.join(base_path, data_dir, id, "events.csv"))
                self.traces = pd.read_csv(os.path.join(base_path, data_dir, id, "positions.csv"))
                self.teams = pd.read_csv(os.path.join(base_path, data_dir, id, "teams.csv"))
            elif "BEPRO" in self.data_dir:
                self.events = pd.read_csv(os.path.join(base_path, data_dir, id, "events.csv"))
                self.events["start_x"] = self.events["trace_start_x"].values
                self.events["start_y"] = self.events["trace_start_y"].values
                self.events["qualifier"] = self.events["qualifier"].apply(
                    lambda s: ast.literal_eval(s) if isinstance(s, str) else s
                )

                # 민호님께서 제공해주신 BEPRO 데이터셋 전처리 방식
                self.events = vaep.bepro.convert_to_actions(self.events, game_id=id)
                self.traces = pd.read_csv(os.path.join(base_path, data_dir, id, "positions.csv"), dtype={'ball_state': str, 'ball_owning_team_id': str})
                self.teams = pd.read_csv(os.path.join(base_path, data_dir, id, "teams.csv"))
            else:
                raise ValueError("only exist BEPRO or DFL path")
            
            # 1.전처리: 공격방향 통일
            #if self.play_left_to_right:
            self.events, self.traces = self._left_right_inversion(self.events, self.traces, id) # 좌우 반전

            # 2. 전처리: 이벤트 데이터 필터링 & unique event_id 생성
            if actionfilter:
                self.events = self.events[self.actionfilter(self.events)].reset_index(drop=True)
            self.events["event_id"] = range(len(self.events))  # event_id: unique identifier for each event

            # 3. 전처리: 트래킹 데이터 필터링: as we did't use ball position, speed either.
            self.traces = self.traces[[col for col in self.traces.columns if not col.startswith("B") and not col.endswith("_speed")]]  

            # 4. 전처리: 이벤ㅌ 시점 별 경기에 뛴 선수 정보 생성
            self.active_players_dict = self._get_active_players()  # if interpolate, active player가 없는 경우에도 위치값이 생성됨에 따라 사전에 추출
            active_players_dict_lst.extend(list(self.active_players_dict.values()))

            # create features and labels
            self._features = self._build_features()
            self._labels = self._build_labels()

            
            if self.model == 'agentimputer':
                self._timestamps = self._build_timestamps()

            # unified feature and label column names
            playerid_to_xid = {row.player_id: f"{row.team[0]}{row.xID:02d}" for row in self.teams.itertuples()}

            if "DFL" in self.data_dir:
                self._features.columns = [f"{playerid_to_xid[col.split('_')[0]]}_{col.split('_')[1]}" for col in self._features.columns] # map player_id to xID
            elif "BEPRO" in self.data_dir:
                self._features.columns = [f"{playerid_to_xid[int(col.split('_')[0])]}_{col.split('_')[1]}" for col in self._features.columns] # map player_id to xID
                self.teams = pd.read_csv(os.path.join(base_path, data_dir, id, "teams.csv"))
            else:
                raise ValueError("only exist BEPRO or DFL path")
                        
            self._features = self._features[sorted(self._features.columns)] # sorted by xID
            self._labels = self._labels[sorted(self._labels.columns)] # sorted by xID

            # sorted by xID: f[0] = xID를 활용하여 정렬한 후 제거
            if self.freeze_frame is not None:
                self.freeze_frame["freeze_frame"] = self.freeze_frame["freeze_frame"].apply(
                    # lambda frame: [f[1:] for f in sorted(frame, key=lambda f: f[0])]
                    lambda frame: [f for f in sorted(frame, key=lambda f: f[0])]
                ) # -> f[0]=x, f[1]=y, f[2]=actor(bool), f[3]=team(bool), f[4]=keeper(bool)

            # 카테고리컬 임베딩 인덱스 저장
            self.categorical_indices = [i for i, col in enumerate(sorted(self.xfns)) if col in config.categorical_features.keys()]

            # create training samples
            for idx, row in enumerate(self.events.itertuples()):
                n_away_players = self.teams[self.teams["team"] == "Away"].shape[0] # 원정 팀의 선수 수를 저장: 동적으로 변경하는 두 팀의 선수 수 정보는 매칭 알고리즘에서 필요함. ex) 11
                valid_player_ids = self.active_players_dict[row.event_id]          # 현재 시점에 뛰는 선수들의 ID를 가져옴. ex) ['H01', 'H02', 'H03', ..., 'A01', 'A02', 'A03']

                # 모델에 입력에 사용되는 정보 추출: feature, label, freeze_frame     
                # 현시점 t 기반으로 t-(window//2)~t+(window//2) 범위의 window를 제작합니다.
                seq_feature_lst, seq_label_lst, seq_freeze_frame_lst = self._create_window(idx, self.window)

                # 3. actor ID: 현재 actor가 홈팀인 경우 추출된 순서에 원정팀의 숫자를 더해서 해당 actor의 인덱스를 구합니다.  
                if playerid_to_xid[row.player_id][0]=='H':
                    actor_global_index=int(playerid_to_xid[row.player_id][-2:]) + n_away_players # 홈 팀 선수이면 10~21사이 인덱스
                else:
                    actor_global_index=int(playerid_to_xid[row.player_id][-2:])                  # ex) 원정 팀 선수이면 0~10사이 인덱스
            
                features_lst.append(torch.stack(seq_feature_lst, dim=0)) # (seq_len, total_players, feature_dim)
                labels_lst.append(torch.stack(seq_label_lst, dim=0))     # (seq_len, total_players, 2 or 4)
                if self.freeze_frame is not None:
                    # freeze_frame: 현재 시점에 뛰는 선수들의 freeze_frame을 생성함
                    freeze_frame_lst.append(torch.stack(seq_freeze_frame_lst, dim=0)) # (seq_len, 22, 5)

                    # freeze_frame_mask: 현재 시점에 뛰는 선수들의 freeze_frame_mask을 생성함 -> loss function or prediction 시에 사용
                    seq_freeze_frame_mask_lst=[]
                    half_window = self.window // 2
                    for i in range(-half_window, half_window + 1):
                        event_idx = min(max(idx + i, 0), len(self.events) - 1) # 인덱스 범위 제한
                        _, freeze_frame_mask, _ = self._create_mask(event_idx) # mask: 실제 경기에 참여한 선수들은 1을 할당하는 마스크: (total_players, 1)
                        
                        # freee_frame과 max_agent를 다르게 설정한 이유?: freeze_frame_mask는 label정보 중 카메라에 포착된 선수들만 필터링하는 목적이므로 label과 차원((config.max_agents, 2 or 4))을 맞춰야함
                        # 반면 freeze_frame은 굳이 max_agents까지 패딩할 필요는 없음.
                        if len(freeze_frame_mask) < config.max_agents:
                            freeze_frame_mask = np.concatenate(
                                [freeze_frame_mask, np.zeros(config.max_agents - len(freeze_frame_mask), dtype=bool)]
                            )
                            
                        seq_freeze_frame_mask_lst.append(torch.FloatTensor(freeze_frame_mask))
                    freeze_frame_mask_lst.append(torch.stack(seq_freeze_frame_mask_lst, dim=0)) # (seq_len, total_players, 1)

                if self.model == 'lstm':
                    # Generate timestamp for AgentImputer Model
                    seq_timestamp_lst=self._create_window_timestamp(idx, self.window) # window의 크기를 입력받아 현시점 t 기반으로 t-(window//2)~t+(window//2) 범위의 window를 제작합니다.
                    timestamps_lst.append(torch.stack(seq_timestamp_lst, dim=0))
                
                # 원정 팀 선수 수를 저장: 동적으로 변경하는 두 팀의 선수 수 정보는 매칭 알고리즘에서 필요함.
                n_away_players_total_lst.append(n_away_players)
                n_away_players_active_lst.append(sum([p[0] == "A" for p in valid_player_ids]))

                # actor 인덱스: 현재 이벤트 시점에 뛰는 선수의 정보를 가져옴
                actor_global_index_lst.append(actor_global_index)                                 # 패딩 포함 전체 선수(config.max_agents) 중 actor 인덱스 (loss 제외)
                actor_valid_index_lst.append(valid_player_ids.index(playerid_to_xid[row.player_id]))  # 패딩 제외 실제 선수 중 actor 인덱스 (예측 결과 활용)

        if len(features_lst) == 0:
            pass
        else:
            self.features_lst = pad_sequence(features_lst).transpose(0, 1) # (W, n_events, N, feature_dim) -> (n_events, W, N, feature_dim)
            self.labels_lst = pad_sequence(labels_lst).transpose(0, 1)  # (W, n_events, N, label_dim) -> (n_events, W, N, label_dim)
            if self.freeze_frame is not None:
                self.freeze_frame_lst = pad_sequence(freeze_frame_lst).transpose(0, 1)      # (W, n_events, N, 5) -> (n_events, W, N, 5)
                self.freeze_frame_mask_lst = pad_sequence(freeze_frame_mask_lst).transpose(0, 1) # (W, n_events, N, 1) -> (n_events, W, N, 1)
            else:
                self.freeze_frame_lst = None
                self.freeze_frame_mask_lst = None

            if self.model == 'lstm':
                self.timestamps_lst = pad_sequence(timestamps_lst).transpose(0, 1)
                self.timestamps_lst = self.timestamps_lst.squeeze(-1)

            if self.use_transform:        
                if self.transform is None:
                    self.transform = Transform(xfn=self.xfns, yfn=self.yfns) 
                    self.transform.fit(self.features_lst, self.labels_lst, self.freeze_frame_lst)

                self.features_lst = self.transform.transform(self.features_lst, None, None)
                self.labels_lst = self.transform.transform(None, self.labels_lst, None)

                if self.freeze_frame is not None:
                    self.freeze_frame_lst = self.transform.transform(None, None, self.freeze_frame_lst)
            else:
                self.transform = None

            print(f"features_lst Shape: {self.features_lst.shape}")  
            print(f"labels_lst Shape: {self.labels_lst.shape}") 
            if self.freeze_frame is not None:
                print(f"freeze_frame Shape: {self.freeze_frame_lst.shape}")
                print(f"freeze_frame mask Shape: {len(self.freeze_frame_mask_lst)}")

            # 결측치로 존재하던 선수들은 0으로 padding value할당
            # 왜 처음부터 0으로 할당을 안하고 결측치를 부여한거야?: 0으로 부여할 시 스케일링에 포함되거나 masking과정에서 복잡하기 때문에 사후 처리
            self.features_lst = [torch.nan_to_num(tensor, nan=0.0) for tensor in self.features_lst]
            if self.freeze_frame is not None:
                self.freeze_frame_lst = [torch.nan_to_num(tensor, nan=0.0) for tensor in self.freeze_frame_lst]

            self.n_away_players_total_lst = n_away_players_total_lst
            self.n_away_players_active_lst = n_away_players_active_lst
            self.actor_valid_index_lst = actor_valid_index_lst
            self.actor_global_index_lst = actor_global_index_lst
            self.active_players_dict_lst = active_players_dict_lst

    def _create_window(self, idx: int, window: int) -> torch.Tensor:
        half_window = window // 2
        seq_feature_lst=[]   
        seq_label_lst=[]
        seq_freeze_frame_lst=[]
        for i in range(-half_window, half_window + 1):
            event_idx = min(max(idx + i, 0), len(self.events) - 1) # 인덱스 범위 제한

            # mask: 현재 시점에 뛰는 선수들의 ID를 가져옴
            mask, freeze_frame_mask, visible_area = self._create_mask(event_idx) # t-window ~ t+window에 대한 mask는 시점별로 다르기 때문에 매 시점마다 mask를 생성함: (seq_len, total_players)

            # 1. feature: 현재 시점(event_idx)에 뛰는 선수들의 feature를 가져옴
            if self.model == "heatmap":
                feature_tensor = np.where(
                    mask[:, None, None], # 2d mask
                    np.stack(self._features.loc[event_idx].to_numpy()), # agent x 68 x 105
                    np.nan
                ).astype(np.float32)  

                # feature assertion: 항상 22명보다 작거나 같아야 한다.
                valid_feat = ~np.isnan(feature_tensor).all(axis=(1, 2))
                assert valid_feat.sum() <= 22, f"[feature] {event_idx=} players={valid_feat.sum()} (>22)"  

                if feature_tensor.shape[0] < config.max_agents:
                    feature_tensor = np.vstack([
                        feature_tensor, 
                        np.zeros((config.max_agents-feature_tensor.shape[0], (feature_tensor.shape[1], feature_tensor.shape[2]))) # (n_agents, 68, 105): 
                    ])
                #feature_tensor = feature_tensor[~np.any(np.isnan(feature_tensor), axis=(1, 2))] # (68, 105)모두가 nan인 경우
            else:
                feature_unique = self._features.shape[1] // (len(self.teams))
                feature_tensor = np.where(
                    mask[:, None],
                    self._features.loc[event_idx].values.reshape(-1, feature_unique),
                    np.nan
                ).astype(np.float32)   

                # feature assertion: 항상 22명보다 작거나 같아야 한다.
                valid_feat = ~np.isnan(feature_tensor).all(axis=1)
                assert valid_feat.sum() <= 22, f"[feature] {event_idx=} players={valid_feat.sum()} (>22)"  

                if feature_tensor.shape[0] < config.max_agents:
                    feature_tensor = np.vstack([
                        feature_tensor, 
                        np.zeros((config.max_agents-feature_tensor.shape[0], feature_unique))
                    ]) # (total_players, Fs)   
                          
            seq_feature_lst.append(torch.FloatTensor(feature_tensor))

            # 2. label: # 뛰지 않은 선수들의 label값에 NaN을 할당합니다.
            label_unique = len(self._labels.columns.str.split("_").str[-1].unique())
            label_tensor = np.where(
                mask[:, None],
                self._labels.loc[event_idx].values.reshape(-1, label_unique),
                float('nan')
            ).astype(np.float32)  
            
            # label assertion: 항상 22명보다 작거나 같아야 한다.
            valid_label = ~np.isnan(label_tensor).all(axis=1)
            assert valid_label.sum() <= 22, f"[label]   {event_idx=} players={valid_label.sum()} (>22)"
            if label_tensor.shape[0] < config.max_agents:
                label_tensor = np.vstack([
                    label_tensor, 
                    np.full((config.max_agents - label_tensor.shape[0], label_unique), np.nan)
                ])
            seq_label_lst.append(torch.FloatTensor(label_tensor))

            # freeze_frame: 현재 시점(event_idx)에 뛰는 선수들의 freeze_frame을 가져옴
            if self.freeze_frame is not None:
                freeze_frame = self.freeze_frame.loc[event_idx].values[0] # (n_agents, 5)

                freeze_frame = np.array([
                    [f[1], f[2], f[3], f[4], f[5]]
                    for f in freeze_frame
                ])
                # visible_area = [
                #     (np.clip(area_x[0], 0, config.field_length), np.clip(area_y[0], 0, config.field_width)) for area_x, area_y in visible_area
                # ]
                # freeze_frame = np.array([
                #     [f[1], f[2], f[3], f[4], f[5], 
                #      visible_area[0][0], visible_area[0][1], visible_area[1][0], visible_area[1][1], 
                #      visible_area[2][0], visible_area[2][1], visible_area[3][0], visible_area[3][1]] 
                #     for f in freeze_frame
                # ])

                # 카메라에 포착되지 않은 선수를 제거(출전하지 않은 선수도 제거)
                freeze_frame = np.where(
                    freeze_frame_mask[:, None],
                    np.array(freeze_frame),
                    np.nan
                ).astype(np.float32)  
                freeze_frame = freeze_frame[~np.any(np.isnan(freeze_frame), axis=1)]

                # 좌표 순서로 정렬: freeze_frame은 ID정보가 없기 때문에 기존 ID순서대로 제공하지 않고 랜덤 shuffle
                freeze_frame = np.array(sorted(freeze_frame, key=lambda x: (x[0], x[1]), reverse=False))

                # freeze_frame assertion: 항상 22명보다 작거나 같아야 한다.
                assert freeze_frame.shape[0] <= 22, f"[freeze] {event_idx=} players={freeze_frame.shape[0]} (>22)"

                if freeze_frame.shape[0] < 22:
                    freeze_frame = np.vstack([
                        freeze_frame, 
                        np.zeros((22 - freeze_frame.shape[0], freeze_frame.shape[-1]))
                    ])

                # 0번째 정보는 정렬을 목적으로 사용했으므로 제거
                seq_freeze_frame_lst.append(torch.FloatTensor(freeze_frame))
                
        return seq_feature_lst, seq_label_lst, seq_freeze_frame_lst
    
    def _create_window_timestamp(self, idx: int, window: int) -> torch.Tensor:
        half_window = window // 2
        seq_feature_lst=[]   
        for i in range(-half_window, half_window + 1):
            event_idx=idx+i
            mask, _, _ = self._create_mask(event_idx) 

            if event_idx < 0:
                event_idx = 0  # 시작 인덱스 이전의 윈도우는 처음 인덱스의 값으로 채움
            elif event_idx >= len(self.events):
                event_idx = len(self.events) - 1  # 이벤트 개수를 넘어서는 경우, 마지막 인덱스의 값으로 채움
            feature_tensor = np.stack(mask * self._timestamps.loc[event_idx].values, axis=0).reshape(-1, 1).astype(np.float32)  
            
            if feature_tensor.shape[0] < config.max_agents:
                feature_tensor = np.vstack([feature_tensor, np.zeros((config.max_agents-feature_tensor.shape[0], 1))]) # (total_players, Fs)                                 
            seq_feature_lst.append(torch.FloatTensor(feature_tensor))

        return seq_feature_lst
    
    def _create_mask(self, idx: int) -> torch.Tensor:
        """
        all_players: 전체 선수 ID 리스트 (ex: teams.keys() or list(teams))
        active_players: 현재 시점에 뛴 선수들 ID 리스트 (player_ids)
        
        return: torch.Tensor of shape (len(all_players),), 1 if active, else 0
        """

        # mask: 이벤트 시점에 경기에 뛴 선수
        all_players = sorted([f"{row.team[0]}{row.xID:02d}" for row in self.teams.itertuples()])
        mask = np.array([True if pid in self.active_players_dict[self.events.at[idx, "event_id"]] else False for pid in all_players]) # ex) (40, 1), (39, 1): 선수가 뛰는 경우 True(1)로 해야 feature*mask이 가능

        # freeze_frame_mask: 이벤트 시점에에 카메라에 포착된 선수
        if self.freeze_frame is not None:
            freeze_frame = self.freeze_frame.loc[idx].values[0]
            visible_area = compute_camera_coverage(np.array([[self.events.at[idx, "start_x"], self.events.at[idx, "start_y"]]]), 
                                                camera_info=(0, -20, 20, 30),
                                                pitch_size=(config.field_length, config.field_width))
            
            if self.play_left_to_right:
                # 공격방향을 통일한 경우 카메라 영역 알고리즘을 수행하기 전에 다시 복원함
                # (x,y)=(f[0], f[1]), actor(bool) = f[3]
                player_pos = np.array([
                    [f[1], f[2]] 
                    if f[0][0] == self.events.at[idx, "team"][0] 
                    else [config.field_length - f[1], config.field_width - f[2]]
                    for f in freeze_frame
                ])
            else:
                player_pos = np.array([
                    [f[1], f[2]] for f in freeze_frame]
                )

            # 이벤트 시점에 경기에 뛰지 않은 선수의 좌표는 np.nan으로 기록되어 있기 때문에 항상 False
            freeze_frame_mask = is_inside(visible_area, player_pos) # [True, False, False,...]

        else:
            freeze_frame_mask = None
            visible_area = None

        return mask, freeze_frame_mask, visible_area
        
    def _left_right_inversion(self, events: pd.DataFrame, positions: pd.DataFrame, match_id: str)-> pd.DataFrame:
        '''
        공격방향을 항상 일치시키는 좌우 반전을 수행합니다.
        홈팀은 left->right, 원정팀은 right->left.
        '''

        # DFL버전

        if "DFL" in self.data_dir:
            x_cols = [col for col in positions.columns if col.endswith('_x')]
            y_cols = [col for col in positions.columns if col.endswith('_y')]
            vx_cols = [col for col in positions.columns if col.endswith('_vx')]
            vy_cols = [col for col in positions.columns if col.endswith('_vy')]
            if match_id in ["DFL-MAT-J03WR9", "DFL-MAT-J03YIY", "DFL-MAT-J03YKB", "DFL-MAT-J03YKY"]:
                # "DFL-MAT-J03WR9" 경기는 전반전에 home이 left->right이므로 후반전 데이터를 좌우 반전시킵니다.
                positions.loc[positions["period_id"] == 2, x_cols] = config.field_length - positions.loc[positions["period_id"] == 2, x_cols].values
                positions.loc[positions["period_id"] == 2, y_cols] = config.field_width - positions.loc[positions["period_id"] == 2, y_cols].values

                positions.loc[positions["period_id"] == 2, vx_cols] = - positions.loc[positions["period_id"] == 2, vx_cols].values
                positions.loc[positions["period_id"] == 2, vy_cols] = - positions.loc[positions["period_id"] == 2, vy_cols].values

                events.loc[events["period_id"] == 2, ["start_x", "related_x"]] = config.field_length - events.loc[events["period_id"] == 2, ["start_x", "related_x"]].values
                events.loc[events["period_id"] == 2, ["start_y", "related_y"]] = config.field_width - events.loc[events["period_id"] == 2, ["start_y", "related_y"]].values
            else:
                # 나머지 모든 경기는 후반전에 home이 left->right이므로 전반전 데이터를 좌우 반전시킵니다.
                positions.loc[positions["period_id"] == 1, x_cols] = config.field_length - positions.loc[positions["period_id"] == 1, x_cols].values
                positions.loc[positions["period_id"] == 1, y_cols] = config.field_width - positions.loc[positions["period_id"] == 1, y_cols].values
          
                positions.loc[positions["period_id"] == 1, vx_cols] = - positions.loc[positions["period_id"] == 1, vx_cols].values
                positions.loc[positions["period_id"] == 1, vy_cols] = - positions.loc[positions["period_id"] == 1, vy_cols].values

                events.loc[events["period_id"] == 1, ["start_x", "related_x"]] = config.field_length - events.loc[events["period_id"] == 1, ["start_x", "related_x"]].values
                events.loc[events["period_id"] == 1, ["start_y", "related_y"]] = config.field_width - events.loc[events["period_id"] == 1, ["start_y", "related_y"]].values

            away_x_cols = [col for col in positions.columns if col.endswith('_x') and col.startswith("A")]
            away_y_cols = [col for col in positions.columns if col.endswith('_y') and col.startswith("A")] 

            away_vx_cols = [col for col in positions.columns if col.endswith('_vx') and col.startswith("A")]
            away_vy_cols = [col for col in positions.columns if col.endswith('_vy') and col.startswith("A")]
            
            # 트래킹 데이터 공격 방향 통일
            positions[away_x_cols] = config.field_length - positions[away_x_cols].values
            positions[away_y_cols] = config.field_width - positions[away_y_cols].values
            positions[away_vx_cols] = - positions[away_vx_cols].values
            positions[away_vy_cols] = - positions[away_vy_cols].values        

            # 이벤트 데이터의 공격 방향 통일
            away_idx = events["team"] == "Away"
            events.loc[away_idx, ["start_x", "related_x"]] = config.field_length - events.loc[away_idx, ["start_x", "related_x"]].values
            events.loc[away_idx, ["start_y", "related_y"]] = config.field_width - events.loc[away_idx, ["start_y", "related_y"]].values
         
        elif "BEPRO" in self.data_dir:
            # BEPRO 버전: 홈팀이 전/후반 상관없이 항상 왼쪽->오른쪽으로 공격하고, 원정팀이 왼쪽<-오른쪽으로 공격하도록 설정
            away_x_cols = [col for col in positions.columns if col.endswith('_x') and col.startswith("A")]
            away_y_cols = [col for col in positions.columns if col.endswith('_y') and col.startswith("A")] 

            away_vx_cols = [col for col in positions.columns if col.endswith('_vx') and col.startswith("A")]
            away_vy_cols = [col for col in positions.columns if col.endswith('_vy') and col.startswith("A")]
            
            # 트래킹 데이터 공격 방향 통일
            positions[away_x_cols] = config.field_length - positions[away_x_cols].values
            positions[away_y_cols] = config.field_width - positions[away_y_cols].values
            positions[away_vx_cols] = - positions[away_vx_cols].values
            positions[away_vy_cols] = - positions[away_vy_cols].values        

            # 이벤트 데이터의 공격 방향 통일
            away_idx = events["team"] == "Away"
            events.loc[away_idx, ["start_x", "related_x", "end_x"]] = config.field_length - events.loc[away_idx, ["start_x", "related_x", "end_x"]].values
            events.loc[away_idx, ["start_y", "related_y", "end_y"]] = config.field_width - events.loc[away_idx, ["start_y", "related_y", "end_y"]].values
        else:
            raise ValueError("only exist BEPRO or DFL path")
        
        return events, positions
    
    # @staticmethodß
    def actionfilter(self, actions: pd.DataFrame) -> pd.Series:
        is_penalty_shootout = actions["period_id"] == 5
        is_excluded_action = actions["type_name"].isin(config.exception_actions)

        # some cases have no player_id or coordinates
        has_player_id = actions["player_id"].notna()
        has_coordinates = actions["start_x"].notna() & actions["start_y"].notna()

        return ~is_penalty_shootout & ~is_excluded_action & has_player_id & has_coordinates
    
    def _build_features(self) -> pd.DataFrame:
        """¥yz¥
        Convert raw data into feature DataFrame using features.py utility.
        """
        # xfns가 list면, 각 함수(str/callable)에 대해 열 목록 추출해서 합침
        df_features = []
        i = 0
        for xfn in self.xfns:
            xfn_callable = self._get_xfn_callable(xfn)
            if xfn in ["freeze_frame"]:
                # freeze_frame정보가 사전에 제공되는 경우 tracking 데이터를 사용하지 않아도 됨.
                feats = xfn_callable(self.events, self.teams, self.traces) 
            else:
                feats = xfn_callable(self.events, self.teams) 

                feats.columns = [f"{col}_{xfn}" for col in feats.columns] # rename: feature name + xfn name
            df_features.append(feats)

        df_features = pd.concat(df_features, axis=1).reset_index(drop=True)

        # freeze_frame 따로 저장
        if "freeze_frame" in self.xfns:
            self.freeze_frame = df_features[["freeze_frame"]]
            df_features = df_features.drop("freeze_frame", axis=1) # freeze_frame은 별도로 저장

        return df_features
    
    def _build_labels(self) -> pd.DataFrame:
        """
        Convert raw data into label DataFrame using labels.py utility.
        """
        # 예시 구현:
        df_labels = []
        for yfn in self.yfns:
            yfn_callable = self._get_yfn_callable(yfn)
            labs = yfn_callable(self.events, self.traces) # label
            df_labels.append(labs)

        return pd.concat(df_labels, axis=1).reset_index(drop=True)

    def _build_timestamps(self) -> pd.DataFrame:
        """
        Convert previous agent times into a scaled DataFrame.

        This method filters the features to extract columns that contain 
        "prevAgentTime" in their names, scales these timestamps using 
        RobustScaler, and returns the scaled timestamps as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the scaled timestamps.
        """
        scaler = RobustScaler()
        timestamps = np.array(self._features.filter(like="prevAgentTime"))
        n_events, n_players = timestamps.shape
        timestamps = timestamps.reshape(-1, 1)
        timestamps = scaler.fit_transform(timestamps)
        timestamps = timestamps.reshape(n_events, n_players)
        timestamps_df = pd.DataFrame(timestamps, columns=[f"{x}_ts" for x in self._features.filter(like="prevAgentTime").columns])
        return timestamps_df

    def _get_xfn_callable(self, xfn):
        """
        xfn이 str이면 features 모듈 내 함수를 가져옴옴
        """
        if isinstance(xfn, str):
            # features.py 안에 동일 이름 함수가 있다고 가정
            try:
                return getattr(features, xfn)
            except AttributeError:
                raise ValueError(f"No feature function found that matches '{xfn}'.")
        else:
            raise ValueError("Invalid xfn type. Must be str or callable.")

    def _get_yfn_callable(self, yfn):
        """
        yfn이 str이면 labels 모듈 내 함수를 가져옴
        """
        if isinstance(yfn, str):
            # labels.py 안에 동일 이름 함수가 있다고 가정
            try:
                return getattr(labels, yfn)
            except AttributeError:
                raise ValueError(f"No labeling function found that matches '{yfn}'.")
        else:
            raise ValueError("Invalid yfn type. Must be str or callable.")

    def __len__(self) -> int:
        # 예시로 features 기준
        return len(self.features_lst)

    def _get_active_players(self):
        '''
        Get active players at each event
        All player's number is 22, but it is not always 22 (e.g., red card).
        Order: Away0~11, Home0~11.
        '''

        tracking_by_period = {
            period: df.reset_index(drop=True)
            for period, df in self.traces.groupby("period_id")
        }

        active_players_dict  = {}
        for period_id, group_events in self.events.groupby("period_id"):
            period_traces = tracking_by_period[period_id]

            for row in group_events.itertuples():
                # cloese_idx: index of the closest time in period_traces
                closest_idx = period_traces["time"].sub(row.time_seconds).abs().idxmin()
                closest_traces = period_traces.loc[[closest_idx]]

                active_players = [col.split("_")[0] for col in closest_traces.dropna(axis=1).columns if col.endswith("_x")]
                active_players_dict[row.event_id] = sorted(active_players)

        return active_players_dict
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Return a single sample (features + labels) at the given index.
        Optionally apply transform.
        """

        sample = {
            "features": self.features_lst[idx], # 15개의 features
            "labels": self.labels_lst[idx], #x,y,vx,vy등 위치좌표와 속도
            "n_away_players_active_lst": self.n_away_players_active_lst[idx], # 그라운드에 있는 원정팀의 수
            "actor_valid_index_lst": self.actor_valid_index_lst[idx], #play를 수행하는 선수의 local index
            "actor_global_index_lst": self.actor_global_index_lst[idx], #play를 수행하는 선수의 global index
            "categorical_indices": self.categorical_indices, # feature에서 분류형 feature의 위치
            "n_away_players_total_lst": self.n_away_players_total_lst[idx] #team sheets에 기록된 원정 팀 선수 수
        }

        if self.freeze_frame is not None:
            sample["freeze_frame"] = self.freeze_frame_lst[idx]
            sample["freeze_frame_mask"] = self.freeze_frame_mask_lst[idx] # freeze_frame에 대한 마스크

        # agentimputer: Time-LSTM 모델을 위한 추가 정보
        if self.model == "agentimputer":
            sample["timestamps"] = self.timestamps_lst[idx]

        return sample

if __name__=="__main__":
    '''
        python imputer/datasets.py \
        --data_dir ./data/DFL \
        --model transformer \
        --xfns agentSide nextAgentTime freeze_frame \
        --yfns coordinates \
        --window 5 \
        --dataset train
    '''
    import argparse
    from datasets import ImputerDataset
    
    parser = argparse.ArgumentParser(description="Train PressingDataset with different models and configurations.")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--model", type=str, choices=["heatmap", "transformer", "lstm"], default="heatmap",
                        help="Choose the model type: heatmap, transformer, or lstm")

    parser.add_argument("--window", type=int, default=5, help="Window size for spatio-temporal models")
    parser.add_argument("--xfns", type=str, nargs="+", default=["freeze_frame"],
                        help="List of feature functions (xfns)")

    parser.add_argument("--yfns", type=str, nargs="+", default=["coordinates"],
                        help="List of label functions (yfns)")

    parser.add_argument("--dataset", type=str, default="train", help="Transform type: train, valid, or test")
    
    args = parser.parse_args()
    data_path = args.data_dir
    game_ids = sorted([name for name in os.listdir(data_path) if name.isdigit()])
    print(game_ids)
    train_game_ids = ['DFL-MAT-J03YKM']#game_ids[:1]
    print(f"train_game_ids: {train_game_ids}")

    train_dataset = ImputerDataset(
        game_ids=train_game_ids,
        data_dir=args.data_dir,
        xfns=args.xfns,
        yfns=args.yfns,
        window=args.window,
        model=args.model,
    )