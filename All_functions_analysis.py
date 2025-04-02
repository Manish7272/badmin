import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import subprocess


def analyze_badminton_video(video_path, model_paths, stats_output_paths):
    # Load YOLO models
    player_model = YOLO(model_paths['player'])
    shuttle_model = YOLO(model_paths['shuttle'])
    shot_model = YOLO(model_paths['shot'])
    pose_model = YOLO(model_paths['pose'])

    # Video setup
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps
    
    output_path = stats_output_paths["output_path"]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Mini-Court properties
    court_width, court_height = 250, 300
    court_x, court_y = frame_width - court_width - 20, 20

    def map_to_mini_court(x, y):
        mini_x = int((x / frame_width) * court_width) + 20
        mini_y = int((y / frame_height) * court_height) + 20
        return mini_x, mini_y
    
    
    # Tracking variables
    prev_player_positions = {}
    prev_shuttle_position = None
    player_speeds = {}
    total_player_speeds = {}
    shuttle_speeds = []
    player_stats = []
    player_shots = {1: [], 2: []}
    frame_count = 0
    player_poses = {1: [], 2: []}

    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        temp_player_positions = {}
        
        # Run detections
        player_results, shuttle_results = player_model(frame), shuttle_model(frame)
        shot_results, pose_results = shot_model(frame), pose_model(frame)

        # Process player detections
        for result in player_results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result.cpu().numpy()
            player_id = int(cls + 1)
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            temp_player_positions[player_id] = (center_x, center_y)

            # Draw player bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {player_id} ({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Compute player speeds
        for player_id, (x, y) in temp_player_positions.items():
            if player_id in prev_player_positions:
                prev_x, prev_y = prev_player_positions[player_id]
                distance = np.linalg.norm([x - prev_x, y - prev_y])
                speed = distance / frame_time
                player_speeds[player_id] = speed
                total_player_speeds.setdefault(player_id, []).append(speed)

        prev_player_positions = temp_player_positions

        # Process shuttle detections
        for result in shuttle_results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result.cpu().numpy()
            shuttle_x, shuttle_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if prev_shuttle_position:
                prev_x, prev_y = prev_shuttle_position
                distance = np.linalg.norm([shuttle_x - prev_x, shuttle_y - prev_y])
                shuttle_speed = distance / frame_time
                shuttle_speeds.append(shuttle_speed)

            prev_shuttle_position = (shuttle_x, shuttle_y)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, "Shuttlecock", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    

        # Process shot detections
        for result in shot_results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result.cpu().numpy()
            shot_type_name = shot_model.names.get(int(cls), "Unknown")  # Handle missing class names safely

            # Identify closest player to the shot
            closest_player_id = None
            min_distance = float("inf")

            shot_center_x, shot_center_y = (x1 + x2) / 2, (y1 + y2) / 2
            for player_id, (px, py) in temp_player_positions.items():
                distance = np.linalg.norm([shot_center_x - px, shot_center_y - py])
                if distance < min_distance:
                    min_distance = distance
                    closest_player_id = player_id
         
            # Save shot for the correct player
            if closest_player_id is not None:
                player_shots[closest_player_id].append([frame_count, shot_type_name])

            # Draw bounding box & label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, f"{shot_type_name} ({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 

        # Process pose detections
        for result in pose_results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result.cpu().numpy()
            pose_name = pose_model.names.get(int(cls), "Unknown")

            # Find the closest player
            closest_player_id = None
            min_distance = float("inf")
            pose_center_x, pose_center_y = (x1 + x2) / 2, (y1 + y2) / 2

            for player_id, (px, py) in temp_player_positions.items():
                distance = np.linalg.norm([pose_center_x - px, pose_center_y - py])
                if distance < min_distance:
                    min_distance = distance
                    closest_player_id = player_id

            # Save pose for the correct player
            if closest_player_id is not None:
                player_poses[closest_player_id].append([frame_count, pose_name])

            # Draw bounding box & label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{pose_name} ({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
        # Store stats
        row = [frame_count]
        for player_id in sorted(total_player_speeds.keys()):
            avg_speed = np.mean(total_player_speeds[player_id])
            row.extend([player_speeds.get(player_id, 0), avg_speed])
        player_stats.append(row)

        # Draw mini-court
        court = np.ones((court_height, court_width, 3), dtype=np.uint8) * 255
        cv2.rectangle(court, (20, 20), (230, 280), (0, 0, 0), 2)
        cv2.line(court, (20, 150), (230, 150), (0, 0, 0), 2)
        for player_id, (x, y) in temp_player_positions.items():
            mini_x, mini_y = map_to_mini_court(x, y)
            color = (0, 255, 0) if player_id == 1 else (0, 0, 255)
            cv2.circle(court, (mini_x, mini_y), 8, color, -1)
        frame[court_y:court_y + court_height, court_x:court_x + court_width] = court

        # Draw stats table at the bottom-right
        table_x, table_y = frame_width - 320, frame_height - 120  # Adjust position to right-bottom
        cv2.rectangle(frame, (table_x, table_y), (frame_width - 10, frame_height - 10), (50, 50, 50), -1)

        # Player statistics text
        stats = [
            f"P1 Speed: {player_speeds.get(1, 0):.2f} px/s", 
            f"P2 Speed: {player_speeds.get(2, 0):.2f} px/s", 
            f"Avg P1: {np.mean(total_player_speeds.get(1, [0])):.2f} px/s", 
            f"Avg P2: {np.mean(total_player_speeds.get(2, [0])):.2f} px/s", 
        ]

        # Draw the text on the overlay
        for i, text in enumerate(stats):
            cv2.putText(frame, text, (table_x + 10, table_y + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Save statistics
    stats_df = pd.DataFrame(player_stats, columns=["Frame"] + [f"Player {i} Speed" for i in sorted(total_player_speeds.keys())] + [f"Avg P{i} Speed" for i in sorted(total_player_speeds.keys())])
    stats_df.to_csv(stats_output_paths['player_stats'], index=False)

    for player in [1, 2]:
        pd.DataFrame(player_shots[player], columns=["Frame", "Shot_Type"]).to_csv(stats_output_paths[f'player{player}_shots'], index=False)
        pd.DataFrame(player_poses[player], columns=["Frame", "Pose"]).to_csv(stats_output_paths[f'player{player}_poses'], index=False)

# Example usage:
# video_path = r"Input_videos\clip 1.mp4"
# video_path = ""

# model_paths = {
#     'player': r"model/players_detection_best.pt",
#     'shuttle': r"model/Shuttlecock_detection_best.pt",
#     'shot': r"model/shotType_detection_best.pt",
#     'pose': r"model/Pose_classification_best.pt"
# }
# stats_output_paths = {
#     "output_path": r"output_video.mp4",
#     "player_stats": r"player_stats.csv",
#     "player1_shots": r"player1_shots.csv",
#     "player2_shots": r"player2_shots.csv",
#     "player1_poses": r"player1_poses.csv",
#     "player2_poses": r"player2_poses.csv"
# }

# analyze_badminton_video(video_path, model_paths, stats_output_paths)




# -------------------------------------------- MP4 to h.264 --------------------

# def convert_to_h264(input_file, output_file):
#     command = [
#         "ffmpeg",
#         "-i", input_file,          # Input file
#         "-c:v", "libx264",         # Video codec: H.264
#         "-preset", "slow",         # Encoding speed/quality
#         "-crf", "23",              # Quality (lower = better, 18-28 is a good range)
#         "-c:a", "aac",             # Audio codec: AAC
#         "-b:a", "128k",            # Audio bitrate
#         output_file
#     ]
    
#     subprocess.run(command, check=True)
#     print(f"Conversion completed: {output_file}")

# # Example usage
# input_mp4 = "output/output_video.mp4"
# output_h264 = "output_video_h264.mp4"

# convert_to_h264(input_mp4, output_h264)
