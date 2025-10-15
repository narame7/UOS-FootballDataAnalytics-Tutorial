import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import imputer.config as config

class Transform:
    def __init__(self, xfn=[], yfn=[]):
        self.xfn = sorted([fn for fn in xfn if fn != "freeze_frame"])
        self.yfn = sorted(yfn)

        # 인덱스 구분
        self.time_indices = [i for i, col in enumerate(self.xfn) if col in config.time_features]
        #self.angle_indices = [i for i, col in enumerate(self.xfn) if col in config.angle_features]

        self.time_scaler = RobustScaler()
        #self.velocity_scaler = RobustScaler()

    def fit(self, X=None, Y=None, freeze_frame_lst=None):
        """훈련 데이터에서 정규화 값 계산"""

        if X is not None:
            Bx, Wx, Nx, Fx = X.shape

            X_center = X[:, Wx // 2, :, :]  # 중앙 timestep 선택 (원본 코드 유지)
            if isinstance(X_center, torch.Tensor):
                X_center = X_center.cpu().numpy()
            # time_features만 선택해 fit
            if len(self.time_indices) != 0:
                self.time_scaler.fit(X_center[..., self.time_indices].reshape(-1, len(self.time_indices)))
            #self.angle_scaler.fit(X_center[..., self.angle_indices].reshape(-1, len(self.angle_indices)))

        if Y is not None:
            Bx, Wx, Nx, Fy = Y.shape
            Y_center = Y[:, Wx // 2, :, :]
            if isinstance(Y_center, torch.Tensor):
                Y_center = Y_center.cpu().numpy()
            Y_reshaped = Y_center.reshape(-1, Fy)

            # if self.yfn == ["velocity"]:
            #     self.velocity_scaler.fit(Y_reshaped)
            # elif self.yfn == ["coordinates", "velocity"]:
            #     self.velocity_scaler.fit(Y_reshaped[..., :2])

        if freeze_frame_lst is not None:
            pass  # 위치 좌표의 min/max는 고정

    def transform(self, X=None, Y=None, freeze_frame=None):
        """훈련된 Scaler를 이용해 데이터를 정규화"""

        if X is not None:
            for i, name in enumerate(self.xfn):
                if name.endswith("X"):
                    X[..., i] /= config.field_length
                elif name.endswith("Y"):
                    X[..., i] /= config.field_width
                elif name.endswith("disttogoal"):
                    X[..., i] /= np.sqrt(config.field_length ** 2 + (config.field_width / 2) ** 2)

            # time_features transform (numpy -> torch)
            if len(self.time_indices) != 0:
                X_time = X[..., self.time_indices]
                X_time_shape = X_time.shape

                X_time_np = X_time.cpu().numpy() if isinstance(X_time, torch.Tensor) else X_time
                X_time_scaled = self.time_scaler.transform(X_time_np.reshape(-1, len(self.time_indices))).reshape(X_time_shape)

                X[..., self.time_indices] = torch.from_numpy(X_time_scaled).to(X.device)

            return X

        if Y is not None:
            if self.yfn == ["coordinates"]:
                Y[..., 0] /= config.field_length
                Y[..., 1] /= config.field_width
            elif self.yfn == ["velocity"]:
                Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
                Y_scaled = self.velocity_scaler.transform(Y_np)
                Y = torch.from_numpy(Y_scaled).to(Y.device) if isinstance(Y, torch.Tensor) else Y_scaled
            elif self.yfn == ["coordinates", "velocity"]:
                Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
                # Y_shape = Y_np[..., :2].shape  # (batch_size, seq_len, num_players, 2)
                # Y_flat = Y_np[..., :2].reshape(-1, 2)  # (batch_size * seq_len * num_players, 2)
                # Y_scaled = self.velocity_scaler.transform(Y_flat)
                # Y_np[..., :2] = Y_scaled.reshape(Y_shape)
                Y_np[..., 2] /= config.field_length
                Y_np[..., 3] /= config.field_width
                Y = torch.from_numpy(Y_np).to(Y.device) if isinstance(Y, torch.Tensor) else Y_np
            else:
                raise ValueError("Not exist:", self.yfn)

            return Y

        if freeze_frame is not None:
            freeze_frame[..., 0] /= config.field_length
            freeze_frame[..., 1] /= config.field_width

        return freeze_frame
