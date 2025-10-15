from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
from scipy.spatial.distance import cdist
import torch.nn.functional as F

import PlayerImputer.imputer.config as config

class Calculator:
    def __init__(self):
        self.global_idx = 0
        self.total_success = 0
        self.total_samples = 0
        self.all_preds = []
        self.all_targets = []
        self.all_global_idx = []
        self.all_local_idx = None  
        self.all_team_n_away = None
        self.all_play_n_away = None 

        self.event_to_accuracy = {}
        self.event_id = 0

    def load_batch_data(self, data_loader):
        """data_loader에서 필요한 데이터를 한 번에 로드"""
        self.all_global_idx = torch.cat([batch["actor_global_index_lst"] for batch in data_loader], dim=0)
        self.all_local_idx = torch.cat([batch["actor_valid_index_lst"] for batch in data_loader], dim=0)
        self.all_team_n_away = torch.cat([batch["n_away_players_total_lst"] for batch in data_loader], dim=0)
        self.all_play_n_away = torch.cat([batch["n_away_players_active_lst"] for batch in data_loader], dim=0)
    
    def match_players(self, preds, targets, event_id, is_away=True):
        """
        예측된 좌표(preds)와 실제 좌표(targets)에 대해 헝가리안 알고리즘을 이용한 최적 매칭 수행.
        
        preds : (현재 away/home에서 뛰고 있는 선수들 수, label_dim)
        targets : (현재 away/home에서 뛰고 있는 선수들 수, label_dim)
        is_away : 현재 데이터가 away팀인지 나타내는 boolean 변수        
        """
        
        local_actor_idx = self.all_local_idx[event_id].item() # 현재 list에서 actor의 idx
        play_n_away=self.all_play_n_away[event_id].item() # 현재 뛰고 있는 away 선수
        
        if preds.shape[-1] == 4:
            preds = preds[:, 2:]  # 마지막 두 개 차원(x, y)만 사용
            targets = targets[:, 2:]
        

        matched_pairs = []

        # 기본적으로 모든 인덱스에 대해서 헝가리안 알고리즘 사용: actor제외
        pred_indices = list(range(preds.shape[0]))
        target_indices = list(range(preds.shape[0]))

        if is_away:
            if local_actor_idx<play_n_away:  #actor가 원정팀일때
                matched_pairs.append((local_actor_idx, local_actor_idx))
                pred_indices.remove(local_actor_idx)
                target_indices.remove(local_actor_idx)
        else: #actor가 홈팀일때
            if local_actor_idx>=play_n_away: #actor가 홈팀일때
                matched_pairs.append((local_actor_idx - play_n_away, local_actor_idx - play_n_away))
                pred_indices.remove(local_actor_idx - play_n_away)
                target_indices.remove(local_actor_idx - play_n_away)


        # 3. actor 제외한 부분 cost 행렬 생성
        cost_matrix = cdist(preds[pred_indices], targets[target_indices], metric="euclidean")

        # 4. 나머지 선수에 대해 헝가리안 알고리즘 적용
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
        for r, c in zip(row_ind, col_ind):
            matched_pairs.append((pred_indices[r], target_indices[c]))

        # actor가 정확하게 매칭되었는지 확인하는 과정
        # for i, j in matched_pairs:
        #     if is_away:
        #         if local_actor_idx<play_n_away:
        #             if i == local_actor_idx:
        #                 if i != j:
        #                     print(i,j, local_actor_idx, play_n_away, matched_pairs)
        #                     raise ValueError("Away actor index should not match target index.")
        #     else:
        #         if i == (local_actor_idx - play_n_away):
        #             if i != j:
        #                 print(i,j, local_actor_idx, play_n_away, matched_pairs)
        #                 raise ValueError("Home actor index should not match target index.")

        success_count = sum(pred_idx == true_idx for pred_idx, true_idx in matched_pairs)

        return matched_pairs, success_count, len(matched_pairs)

    def process_team(self, preds, targets, event_id):
        """
        팀별 매칭을 수행하는 함수 (Away팀 & Home팀)
        """
        
        '''
        targets.shape : torch.Size([40, 2])
        preds.shape : torch.Size([40, 2])
        team_n_away : team_sheets에 기록된 총 away 선수
        play_n_away : 현재 뛰고 있는 away 선수
        
        '''
        team_n_away=self.all_team_n_away[event_id]                
        away_preds = preds[:team_n_away]     # (W, N, label_dim)
        away_targets = targets[:team_n_away] # (W, N, label_dim)
        valid_indices = ~torch.isnan(away_targets).all(dim=1) # target이 NaN인 선수들은 계산에서 제외

        away_targets, away_preds = away_targets[valid_indices], away_preds[valid_indices]
        away_match_pairs, away_success, away_samples = self.match_players(away_preds, away_targets, event_id, is_away=True)
        

        home_preds = preds[team_n_away:]
        home_targets = targets[team_n_away:]
        valid_indices = ~torch.isnan(home_targets).all(dim=1)
        home_targets, home_preds = home_targets[valid_indices], home_preds[valid_indices]
        home_match_pairs, home_success, home_samples = self.match_players(home_preds, home_targets, event_id, is_away=False)

        self.total_success += home_success+away_success
        self.total_samples += home_samples+away_samples

        self.event_to_accuracy[self.event_id] = home_success+away_success
        self.event_id += 1
        
        return away_match_pairs,home_match_pairs
    
    def process_preds(self, preds, targets, event_id):
        if preds.shape[-1] == 4:
            preds = preds[:, 2:]  # 마지막 두 개 차원(x, y)만 사용
            targets = targets[:, 2:]
            
        team_n_away=self.all_team_n_away[event_id]                
        away_preds = preds[:team_n_away]     # (W, N, label_dim)
        away_targets = targets[:team_n_away] # (W, N, label_dim)
        valid_indices = ~torch.isnan(away_targets).all(dim=1) # target이 NaN인 선수들은 계산에서 제외
        away_targets, away_preds = away_targets[valid_indices], away_preds[valid_indices]

        home_preds = preds[team_n_away:]
        home_targets = targets[team_n_away:]
        valid_indices = ~torch.isnan(home_targets).all(dim=1)
        home_targets, home_preds = home_targets[valid_indices], home_preds[valid_indices]

        return away_targets, away_preds, home_targets, home_preds

    def get_accuracy(self):
        """
        전체 매칭 정확도 반환
        """
        return self.total_success / self.total_samples if self.total_samples > 0 else 0.0
    