import math
import json
import numpy as np
import torch
from shapely.geometry import Point, Polygon

def load_single_json(file_path):
    """
    Loads and parses a single JSON file from the specified path.

    Args:
        file_path (str): The path to the JSON file to load.

    Returns:
        dict or list or None: The parsed Python object on success.
                              Returns None if the file is not found or
                              is not valid JSON.
    """
    try:
        # Open the file in read mode ('r') with UTF-8 encoding.
        # Using 'with' ensures the file is automatically closed after use.
        with open(file_path, 'r', encoding='utf-8') as f:
            # json.load() parses the JSON data from the file object.
            data = json.load(f)
            #print(f"Successfully loaded file: '{file_path}'")
            return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Check format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{file_path}': {e}")
        return None

def load_jsonl(file_path):
    """
    Loads data from a JSON Lines (.jsonl) file.

    Each line in the file is expected to be a valid JSON object.
    Lines that are empty or cannot be parsed as JSON will be skipped with a warning.

    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        list: A list containing the Python objects parsed from each valid JSON line.
              Returns an empty list if the file is not found or contains no valid JSON lines.
    """
    data = [] # To store the parsed JSON objects from each line
    try:
        # Open the file in read mode ('r') with UTF-8 encoding.
        # 'with' ensures the file is closed automatically.
        with open(file_path, 'r', encoding='utf-8') as f:
            # Iterate through each line in the file.
            # enumerate adds line numbers (starting from 1) for better error reporting.
            for line_number, line in enumerate(f, 1):
                # Remove leading/trailing whitespace (including the newline character \n)
                processed_line = line.strip()

                # Skip empty lines
                if not processed_line:
                    continue

                try:
                    # Parse the current line (which is a string) into a Python object.
                    # Use json.loads() for parsing a string, not json.load().
                    parsed_object = json.loads(processed_line)
                    data.append(parsed_object)
                except json.JSONDecodeError:
                    # Handle lines that are not valid JSON.
                    print(f"Warning: Skipping line {line_number} in '{file_path}' due to JSON decoding error.")
                    # Optional: Print the problematic line for debugging
                    # print(f"         Problematic line content: {processed_line[:100]}...")
                except Exception as e:
                    # Handle any other unexpected errors during line processing
                    print(f"Warning: An unexpected error occurred processing line {line_number} in '{file_path}': {e}. Skipping line.")

    except FileNotFoundError:
        # Handle the case where the file itself doesn't exist.
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        # Handle other potential errors during file opening or reading (outside the line loop).
        print(f"An error occurred while reading the file '{file_path}': {e}")

    # Return the list of successfully parsed objects.
    return data

def compute_camera_coverage(ball_pos: np.ndarray, camera_info=(0, -20, 20, 30), pitch_size=(108, 72)):
    '''
    ball_pos: [bs, time, 2]: x, y
    camera_info: [x, y, z, fov]: x, y, z: camera position in the field
    pitch_size: [length, width]: field size

    Returns:
    vertices(tuple): (poly_loc0, poly_loc1, poly_loc2, poly_loc3)
    poly_loc(tuple): ([[x0, x1, x2]], [[y0, y1, y2]]) # 매 시점 별 좌표
    '''
    # Camera info
    camera_x = camera_info[0]
    camera_y = camera_info[1]
    camera_z = camera_info[2]
    camera_fov = camera_info[3]

    # Camera settings
    camera_x = pitch_size[0] / 2
    camera_xy = np.array([camera_x, camera_y])
    camera_height = camera_z
    camera_ratio = (16, 9)

    camera_fov_x = math.radians(camera_fov)
    camera_fov_y = math.radians(camera_fov / camera_ratio[0] * camera_ratio[1])

    ball_right = ball_pos[..., 0] > pitch_size[0] / 2

    # Camera-ball angle
    ball_camera_dist = np.linalg.norm(ball_pos - camera_xy, axis=-1) #torch.norm(ball_pos - camera_xy, dim=-1)  # [bs, time]
    camera_ball_angle = math.pi / 2 - np.arctan(
        abs(camera_xy[1] - ball_pos[..., 1]) / abs(camera_xy[0] - ball_pos[..., 0] + 1e-8)
    )
    camera_ball_angle_y = np.arctan(camera_height / ball_camera_dist)

    front_dist = camera_height / np.tan(camera_ball_angle_y + camera_fov_y / 2)
    rear_dist = camera_height / np.tan(camera_ball_angle_y - camera_fov_y / 2)
    # front_dist = camera_height / math.tan(camera_ball_angle_y + camera_fov_y / 2)
    # rear_dist = camera_height / math.tan(camera_ball_angle_y - camera_fov_y / 2)

    front_ratio = front_dist / ball_camera_dist
    rear_ratio = rear_dist / ball_camera_dist

    camera_fov = math.radians(camera_fov)  # in degree

    # Create a Polygon from the coordinates
    # ball_fov_dist_x = (ball_camera_dist * math.tan(camera_fov_x / 2)) * math.cos(camera_ball_angle)
    # ball_fov_dist_y = (-1)  (ball_right) * (ball_camera_dist * math.tan(camera_fov_x / 2))
    # * math.sin(camera_ball_angle)
    # ball_fov_dist_y = (ball_camera_dist * math.tan(camera_fov / 2)) * math.sin(camera_ball_angle)

    ball_camera_close_dist = ball_camera_dist * front_ratio
    ball_camera_far_dist = ball_camera_dist * rear_ratio

    sign_y = (-1) ** ball_right
    ball_fov_close_dist_x = (ball_camera_close_dist * np.tan(camera_fov_x / 2)) * np.cos(camera_ball_angle)
    ball_fov_close_dist_y = sign_y * (ball_camera_close_dist * np.tan(camera_fov_x / 2)) * np.sin(camera_ball_angle)
    ball_fov_far_dist_x = (ball_camera_far_dist * np.tan(camera_fov_x / 2)) * np.cos(camera_ball_angle)
    ball_fov_far_dist_y = sign_y * (ball_camera_far_dist * np.tan(camera_fov_x / 2)) * np.sin(camera_ball_angle)

    front_ratio = front_ratio[..., None]
    rear_ratio = rear_ratio[..., None]

    close_fov_center_point = ball_pos * (front_ratio) + camera_xy * (1 - front_ratio)
    far_fov_center_point = ball_pos * rear_ratio + camera_xy * (-rear_ratio + 1)

    poly_loc0 = (
        far_fov_center_point[..., 0] - ball_fov_far_dist_x,
        far_fov_center_point[..., 1] - ball_fov_far_dist_y,
    )
    poly_loc1 = (
        far_fov_center_point[..., 0] + ball_fov_far_dist_x,
        far_fov_center_point[..., 1] + ball_fov_far_dist_y,
    )
    poly_loc2 = (
        close_fov_center_point[..., 0] + ball_fov_close_dist_x,
        close_fov_center_point[..., 1] + ball_fov_close_dist_y,
    )
    poly_loc3 = (
        close_fov_center_point[..., 0] - ball_fov_close_dist_x,
        close_fov_center_point[..., 1] - ball_fov_close_dist_y,
    )

    vertices = (poly_loc0, poly_loc1, poly_loc2, poly_loc3)

    return vertices

def is_inside(polygon_vertices, player_positions):
    """
    polygon_vertices: List of (x, y) tuples — e.g., visible_area
    player_positions: numpy array of shape (N, 2)
    Returns: Boolean numpy array of shape (N,) indicating whether each point is inside the polygon
    """
    polygon = Polygon(polygon_vertices)
    return np.array([
        polygon.contains(Point(x, y)) for x, y in player_positions
    ])

if __name__ == "__main__":
    ball_pos = np.array([[0, 0], [10, 10], [20, 20]])
    camera_info = (0, -20, 20, 30)
    pitch_size = (108, 72)
    vertices = compute_camera_coverage(ball_pos, camera_info, pitch_size)
    print(vertices)