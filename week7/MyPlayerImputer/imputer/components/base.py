"""Model architectures."""

import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from gplearn.genetic import SymbolicClassifier
from rich.progress import track
from sklearn.model_selection import cross_val_score, train_test_split
from torch.utils.data import DataLoader, Subset, random_split
import mlflow
from imputer.datasets import ImputerDataset
import imputer.config as config

from pytorch_lightning.loggers import TensorBoardLogger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class exPressComponent(ABC):
    """Base class for all components."""

    component_name = "Default"

    def _get_cell_indexes(self, x, y):
        # x_bin = torch.clamp(x, 0, 25 - 1).to(torch.uint8)
        # y_bin = torch.clamp(y, 0, 25 - 1).to(torch.uint8)
        x_bin = torch.clamp(x, 0, config.field_length - 1).to(torch.uint8)
        y_bin = torch.clamp(config.field_width - y, 0, config.field_width - 1).to(torch.uint8)

        return x_bin.item(), y_bin.item() # convert tensor to integer: tensor index is a problem (broadcasting)
    
    @abstractmethod
    def train(self, train_dataset: Callable, valid_dataset: Callable) -> Optional[float]:
        pass

    @abstractmethod
    def test(self, dataset: Callable) -> Dict[str, float]:
        pass

    def _get_metrics(self, y_true, y_hat):
        return {}

    @abstractmethod
    def predict(self, dataset: Callable) -> pd.Series:
        pass

    def save(self, path: Path):
        pickle.dump(self, path.open(mode="wb"))

    def load(self, path: Path):
        print(f"🔥 Loading model from {path}")
        checkpoint = torch.load(path, map_location="cpu")  # ✅ 체크포인트 로드
        self.model.load_state_dict(checkpoint["state_dict"])  # ✅ 가중치 적용
        
class exPressPytorchComponent(exPressComponent):
    """Base class for a PyTorch-based component."""

    def __init__(self, model, params):

        super().__init__()
        self.model = model
        self.params = params
        self.save_path = params["save_path"]
        
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=self.save_path, filename="best_model", 
                                              **params["ModelCheckpoint"])
        early_stop_callback = EarlyStopping(monitor="val_loss", **params["EarlyStopConfig"])

        run = mlflow.active_run()
        if run is None:
            mlflow.start_run()
        
        # # Init lightning trainer
        self.trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], 
                                  enable_progress_bar=True,
                                  logger=True,
                                  devices=[3],
                                  **self.params["TrainerConfig"])

        mlflow.end_run()

    def train(self, train_dataset, valid_dataset) -> Optional[float]:
        # Load data
        print("Generating datasets...")

        train_dataloader = DataLoader(train_dataset, shuffle=True, **self.params["DataConfig"])
        val_dataloader = DataLoader(valid_dataset, shuffle=False, **self.params["DataConfig"])
        
        # run = mlflow.active_run()
        # if run is None:
        #     mlflow.start_run()

        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
   
        return None
        
    def test(self, dataset) -> Dict[str, float]:
        """모델 평가 (MSE Loss 반환)"""
        dataloader = DataLoader(dataset, shuffle=False, **self.params["DataConfig"])
        outputs = self.trainer.predict(self.model, dataloaders=dataloader, ckpt_path="best")

        preds = torch.cat(outputs, dim=0).cpu().numpy()  # `torch.cat()`으로 리스트 -> Tensor 변환
        targets = np.array([data["labels"] for data in dataset])  # 정답 데이터 수집

        mse_loss = np.mean((preds - targets) ** 2)

        return {"mse_loss": mse_loss}

    def predict(self, dataset) -> pd.Series:
        """모델 예측 (x, y 좌표 반환)"""
        dataloader = DataLoader(dataset, shuffle=False, **self.params["DataConfig"])
        outputs = self.trainer.predict(self.model, dataloaders=dataloader)

        preds = torch.cat(outputs, dim=0).cpu().numpy()  # 리스트 -> NumPy 배열 변환

        return pd.Series(preds)