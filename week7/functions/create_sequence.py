import pandas as pd

# 제공된 설정값들은 그대로 사용합니다.
sequence_starting_candidates = {
    'Pass': [None], 'Pass_Freekick': [None], 'Carry': [None], 'Dribble': [None],
    'Cross': [None], 'Throw-In': [None], 'Pass_Corner': [None],
    'Goal Kick': [None], 'Recovery': [None], 'Duel': ['Successful']
}
sequence_ending_candidates = {
    'Pass': ['Unsuccessful'], 'Dribble': [None], 'Duel': ['Unsuccessful'],
    'Cross': ['Unsuccessful'], 'Throw-In': ['Unsuccessful'], 'Clearance': [None],
    'Interception': [None], 'Out': [None], 'Error': [None], 'Foul': [None],'Shot': [None],
    'Offside': [None], 'Save': [None], 'Tackle': ['Successful'], 'Block': [None],
    'Goal': [None], 'Goal Kick': ['Unsuccessful']
}
set_piece_events = ['Pass_Freekick', 'Pass_Corner', 'Throw-In', 'Goal Kick']

class SequenceAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.sequences_df = None

    def _check_starting(self, row):
        event_type = row['type_name']
        result = row['result_name']
        if event_type in sequence_starting_candidates:
            allowed_results = sequence_starting_candidates[event_type]
            return allowed_results == [None] or result in allowed_results
        return False

    def _check_ending_candidate(self, row):
        event_type = row['type_name']
        result = row['result_name']
        
        # Intervention은 아래 특별 로직에서만 처리하므로 여기서 제외
        if event_type in ['Out']:
            return True
        if event_type == 'Tackle' and result == 'Successful':
            return True
        if event_type in sequence_ending_candidates:
            allowed_results = sequence_ending_candidates[event_type]
            return allowed_results == [None] or result in allowed_results
        return False

    def _create_sequences(self) -> pd.DataFrame:
        df_copy = self.df.copy().sort_values(by=['period_id', 'action_id' if 'action_id' in  self.df.columns else df.index]).reset_index(drop=True)

        df_copy['sequence_id'] = -1
        df_copy['is_sequence_starting'] = False
        df_copy['is_sequence_ending'] = False
        current_sequence = 0
        in_sequence = False
        prev_period = None


        force_new_sequence = False
        for i in range(len(df_copy)):
            row = df_copy.iloc[i]
            period_changed = prev_period is not None and row['period_id'] != prev_period
            prev_period = row['period_id']

            is_starting = self._check_starting(row)
            is_ending_candidate = self._check_ending_candidate(row)

            # 'Block'에 대한 기존 특별 규칙
            if row['type_name'] == 'Block':
                if i + 1 < len(df_copy):
                    next_row = df_copy.iloc[i + 1]
                    if row['team_name'] == next_row['team_name']:
                        is_ending_candidate = True
                    else:
                        is_ending_candidate = False
                else:
                    is_ending_candidate = True
            
            # [신규 규칙] Intervention에 대한 새로운 규칙 적용
            elif row['type_name'] == 'Intervention':
                if i + 1 < len(df_copy):
                    next_row = df_copy.iloc[i + 1]
                    
                    # 조건 1: 팀이 다른가?
                    # teams_are_different = (row['team_name'] != next_row['team_name'])
                    # 조건 2: 다음 이벤트가 시퀀스 시작 이벤트인가?
                    next_is_starting = self._check_starting(next_row)

                    # 두 조건이 모두 참일 때만 종료 이벤트로 간주
                    if next_is_starting:
                        is_ending_candidate = True
                    else:
                        is_ending_candidate = False
                else:
                    # 데이터의 마지막 이벤트가 Intervention이면 종료
                    is_ending_candidate = True

            if period_changed:
                in_sequence = False
                force_new_sequence = False
            
            if (is_starting and not in_sequence) or force_new_sequence:
                current_sequence += 1
                in_sequence = True
                df_copy.loc[i, 'is_sequence_starting'] = True
                force_new_sequence = False

            if in_sequence:
                df_copy.loc[i, 'sequence_id'] = current_sequence

            next_is_ending_candidate = False
            if is_ending_candidate and i + 1 < len(df_copy):
                next_row = df_copy.iloc[i + 1]
                next_is_ending_candidate = self._check_ending_candidate(next_row)

            should_end_now = is_ending_candidate and not next_is_ending_candidate

            if in_sequence and should_end_now:
                df_copy.loc[i, 'is_sequence_ending'] = True
                in_sequence = False
                force_new_sequence = True

        self.sequences_df = df_copy

        
    def find_counter_attacks(
        self,
        # 논문의 규칙에 기반한 파라미터들
        directionality_threshold: float = 0.75,
        min_forward_dist_m: float = 16.0,
        max_duration_sec: float = 14.0,
        start_x_proportion: float = 0.5 # 수비 진영 (하프라인)
    ):
        """
        논문에서 제시한 5가지 규칙을 기반으로 역습 시퀀스를 찾습니다.
        """
        if self.sequences_df is None:
            self._create_sequences()

        # --- 1단계: 모든 공격 시퀀스(possession sequence) 추출 ---
        # [설명] groupby의 결과는 (그룹 이름, 그룹 DataFrame) 튜플이므로 두 변수로 받습니다.
        possession_sequences = self.sequences_df.groupby('sequence_id')
        
        # --- 2단계: 각 시퀀스가 역습 규칙을 만족하는지 검사 ---
        counter_attack_sequences = []
        first_sequence_ids = set(self.sequences_df.groupby('period_id')['sequence_id'].min())

        for sequence_id, sequence_df in possession_sequences:
            # [설명] 시퀀스가 비어있는 경우를 대비한 방어 코드
            if sequence_df.empty:
                continue
            first_event = sequence_df.iloc[0]
            # --- Rule 0: 각 Period 첫번째 시퀀스 제외 ---

            if sequence_id in first_sequence_ids:
                continue
            # [수정] 시퀀스의 첫 이벤트를 기준으로 공격 팀과 공격 방향을 결정합니다.
            #       player_code 대신 함수 인자로 받은 home_team_id를 사용합니다.
            sequence_starting_team_id = first_event['player_code'][0]
            attack_direction = 'RIGHT' if sequence_starting_team_id == 'H' else 'LEFT'
            # --- Rule 1: 시작 위치 검사 ---        
            # 첫 이벤트의 유형이 set_piece_events에 포함되지 않으면 오픈플레이 턴오버입니다.
            is_open_play_turnover = first_event['type_name'] not in set_piece_events

            # 공격 방향에 따라 수비 진영을 판단합니다. (피치 크기 105m 기준)
            if attack_direction == 'RIGHT':
                is_in_defensive_half = first_event['start_x'] < (105 * start_x_proportion)
            else: # attack_direction == 'LEFT'
                is_in_defensive_half = first_event['start_x'] > (105 * start_x_proportion)
            # [설명] 두 조건(오픈플레이 & 수비진영 시작)을 모두 만족하지 않으면 다음 시퀀스로 넘어갑니다.
            if not (is_open_play_turnover and is_in_defensive_half):
                continue

            # --- Rule 2: 세트피스 제외 검사 ---
            if sequence_df['type_name'].isin(set_piece_events).any():
                continue

            # --- Rule 3: 공격 방향성 검사 ---
            # forward_distance, total_distance = # ... (공의 총 이동거리와 순수 전진거리 계산)
            if attack_direction == 'RIGHT':
                forward_distance = sequence_df['end_x'].iloc[-1] - sequence_df['start_x'].iloc[0]
                total_distance = sequence_df['end_x'].iloc[-1] - sequence_df['start_x'].iloc[0]
            else: # attack_direction == 'LEFT'
                forward_distance = sequence_df['start_x'].iloc[0] - sequence_df['end_x'].iloc[-1]
                total_distance = sequence_df['start_x'].iloc[0] - sequence_df['end_x'].iloc[-1]
            if total_distance <= 0:
                continue
            directionality = forward_distance / total_distance
            if directionality < directionality_threshold:
                continue

            # --- Rule 4: 전진 거리 검사 ---
            if forward_distance < min_forward_dist_m:
                continue

            # --- Rule 5: 지속 시간 검사 ---
            duration = sequence_df['time'].max() - sequence_df['time'].min()
            if duration > max_duration_sec:
                continue

            # --- 모든 규칙 통과 시 ---
            start_frame = sequence_df['frame_id'].min()
            end_frame = sequence_df['frame_id'].max()
            counter_attack_sequences.append((start_frame, end_frame))
            
        return counter_attack_sequences