import imageio
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from functions.pitch_control_prep import PitchControlCalculator  # 가정
import os
import datatools.refactored_PC as RPC
import pandas as pd


class makeVideo:
    def __init__(self, pitch_calculator: PitchControlCalculator):
        self.pitch_calculator = pitch_calculator

    # 이미지 프레임들을 동영상으로 변환하는 헬퍼 함수
    def _create_gif_from_frames(self, frames_dir, output_path="animation.gif", fps=30):
        img_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not img_files:
            print("에러: 이미지가 없습니다.")
            return

        images = [imageio.imread(img) for img in img_files]
        imageio.mimsave(os.path.join(frames_dir, output_path), images, fps=fps)
        print(f"GIF 파일 생성 완료: {output_path}")

    def _create_mp4_from_frames(self, frames_dir, output_path="animation.mp4", fps=30):
        """이미지 프레임들을 MP4 동영상으로 변환합니다."""
        img_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not img_files:
            print("에러: 이미지가 없습니다.")
            return

        # 'get_writer'를 사용하여 동영상 파일을 엽니다.
        writer = imageio.get_writer(os.path.join(frames_dir, output_path), fps=fps)

        # 각 이미지 파일을 동영상에 추가합니다.
        for img_path in img_files:
            image = imageio.imread(img_path)
            writer.append_data(image)

        writer.close()  # 파일 쓰기를 완료합니다.
        print(f"MP4 파일 생성 완료: {output_path}")

    def make_animation(
        self,
        melted_df: pd.DataFrame,
        start_frame: int,
        end_frame: int,
        output_filename: str = "animation.gif",
        attacking_only=False,
        attacking_penalty_area_only=True,
        fps: int = 25,
        stride: int = 1,
    ):
        """
        지정된 프레임 구간의 피치 컨트롤 애니메이션을 생성합니다.

        Args:
            start_frame (int): 애니메이션 시작 프레임 ID.
            end_frame (int): 애니메이션 종료 프레임 ID.
            tracking_df (pd.DataFrame): 전체 트래킹 데이터.
            home_mapping (dict): 홈팀 선수 매핑 정보.
            output_filename (str): 저장할 영상 파일 이름.
            fps (int): 영상의 초당 프레임 수.
        """
        base_dir = "animation"
        os.makedirs(base_dir, exist_ok=True)
        existing = [int(d) for d in os.listdir(base_dir) if d.isdigit()]
        next_idx = max(existing) + 1 if existing else 1
        frames_dir = os.path.join(base_dir, str(next_idx))
        os.makedirs(frames_dir, exist_ok=True)

        print(f"👉 이번 실행은 {frames_dir} 폴더에 저장됩니다.")

        frame_ids = range(start_frame, end_frame + 1, stride)
        print(f"총 {len(frame_ids)}개의 프레임 이미지를 생성합니다...")

        for i, frame_id in enumerate(frame_ids):
            # 1. 해당 프레임의 피치 컨트롤 맵과 Figure, Axis 객체를 가져옴
            try:
                PPCFa, ax, _ = self.pitch_calculator.calculate_model_pitch_control(
                    melted_df=melted_df,
                    frame_id=frame_id,
                    attacking_only=attacking_only,
                    attacking_penalty_area_only=attacking_penalty_area_only,
                )
                fig = ax.figure
            except Exception as e:
                print(f"프레임 {frame_id} 처리 중 오류 발생: {e}")
                plt.close("all")  # 오류 발생 시 열려있는 figure 닫기
                continue

            # 4. 이미지 파일로 저장
            filepath = os.path.join(frames_dir, f"frame_{i:05d}.png")
            fig.savefig(filepath)
            plt.close(fig)  # 메모리 누수 방지를 위해 필수

        # 5. 저장된 이미지들을 동영상으로 변환
        final_fps = int(fps / stride)
        # self._create_gif_from_frames(frames_dir, fps=final_fps)
        self._create_mp4_from_frames(frames_dir, fps=final_fps)
