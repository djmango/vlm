# try to couple videos and devents
import boto3
import psycopg2
import ffmpeg
import io
import os
import numpy as np
import cv2
import pyarrow as pa
import pyarrow.parquet as pq

from botocore.exceptions import ClientError
from psycopg2 import OperationalError
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List
from uuid import UUID
from tqdm import tqdm

@dataclass
class Devent:
    id: UUID
    session_id: UUID
    recording_id: UUID
    mouse_action: Optional[str]
    scroll_action: Optional[str]
    mouse_x: int
    mouse_y: int
    event_timestamp: datetime
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    keyboard_action: Optional[str]

@dataclass
class DataRow:
    frame: bytes  # JPEG image data
    mouse_action: Optional[str]
    mouse_x: int
    mouse_y: int
    scroll_action: Optional[Tuple[int, int]]
    keyboard_action: Optional[Tuple[str, str]]

# AWS S3 client
s3 = boto3.client('s3')
s3 = boto3.client('s3',
    aws_access_key_id="AKIAZQ3DUAZ7FYVEUYLT",
    aws_secret_access_key="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"
)
s3_bucket = 'sidekick-videos0'

def list_folders(bucket):
    try:
        response = s3.list_objects(Bucket=bucket, Delimiter='/')
        for prefix in response.get('CommonPrefixes', []):
            print(f"Folder: {prefix['Prefix']}")
    except ClientError as e:
        print(f"Error listing folders: {e}")

def s3_get_recording(session_id, timestamp):
    try:
        # List all objects in the session folder
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=f"{session_id}/")
        
        # Filter and sort video objects
        video_objects = sorted([
            obj for obj in response.get('Contents', [])
            if obj['Key'].endswith('.mp4')
        ], key=lambda x: int(x['Key'].split('/')[-1].split('.')[0]))
        
        # Find the appropriate video
        target_video_start = 0
        target_video = None
        for video in video_objects:
            video_start = int(video['Key'].split('/')[-1].split('.')[0])
            video_end = video_start + 30  # Assuming 30-second videos
            if video_start <= timestamp.timestamp() <= video_end:
                target_video_start = video_start
                target_video = video
                break
        
        if not target_video:
            raise ValueError(f"No suitable video found for timestamp {timestamp}")
        # Download the video to memory
        print(target_video['Key'])
        f = io.BytesIO()
        s3.download_fileobj(s3_bucket, target_video['Key'], f)
        
        # Save f as a .mpf file
        with open('temp_video.mp4', 'wb') as mpf_file:
            mpf_file.write(f.getvalue())
        f = 'temp_video.mp4'
        return f, target_video_start
    except Exception as e:
        print(f"Error in s3_get_recording: {e}")
        return None

def get_video_dimensions(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        return width, height
    except ffmpeg.Error as e:
        print(f"Error probing video: {e.stderr.decode()}")
        return None

def extract_img(recording, timestamp, fps=10):
    try:
        f, start_time = recording
        
        time_offset = timestamp.timestamp() - start_time
        
        output_filename = 'temp_frame.bmp'
        (
            ffmpeg
            .input(f, ss=time_offset)
            .filter('select', 'gte(n,0)')
            .output(output_filename, vframes=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        # Read the BMP file using OpenCV
        img = cv2.imread(output_filename)
        if img is None:
            print(f"Error: Unable to read the frame from {output_filename}")
            return None
        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"Error in extract_img: {e}")
        return None

def extract_actions(row: Devent):
    def parse_tuple(value):
        if isinstance(value, str):
            return tuple(map(lambda x: int(x) if x.strip('-').isdigit() else x, value.strip('()').split(',')))
        return value
    return (
        row.mouse_action,
        row.mouse_x,
        row.mouse_y,
        parse_tuple(row.scroll_action),
        parse_tuple(row.keyboard_action)
    )
# PostgreSQL connection function
def connect_to_postgres():
    try:
        conn = psycopg2.connect(
            "postgresql://idb_owner:R5eM6OLBcmhP@ep-still-lake-a20xdz71.eu-central-1.aws.neon.tech/idb?sslmode=require"
        )
        return conn
    except OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def get_unique_session_ids(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT session_id FROM devents")
            return [row[0] for row in cur.fetchall()]
    except psycopg2.Error as e:
        print(f"Error fetching unique session_ids: {e}")
        return []

def get_all_session_rows(conn, session_id):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM devents WHERE session_id = %s", (session_id,))
            rows = cur.fetchall()
            print(rows[0])
            return [Devent(*row) for row in rows]
    except psycopg2.Error as e:
        print(f"Error fetching rows for session_id {session_id}: {e}")
        return []

def process_session(session_id, rows : List[Devent], save_dir='data'):
    print(f"Processing session {session_id} with {len(rows)} rows")
    rows = sorted(rows, key=lambda x: x.event_timestamp, reverse=False)
    parquet_rows = []
    curr_recording_id = rows[0].recording_id
    curr_recording = None #a 30 sec video objNon
    for i, row in enumerate(rows):
        recording_id = row.recording_id
        timestamp = row.event_timestamp
        print(f"Processing row {i+1}/{len(rows)}: recording_id={recording_id}, timestamp={timestamp}")
        if recording_id != curr_recording_id or curr_recording == None:
            #print(f"New recording detected. Fetching recording for session {session_id}, timestamp {timestamp}")
            curr_recording = s3_get_recording(session_id, timestamp)
            curr_recording_id = recording_id 
            #print(f"Updated curr_recording_id to {curr_recording_id}")

        if curr_recording is not None:
            img, actions = extract_img(curr_recording, timestamp), extract_actions(row)
            #print(f"Extracted image shape: {img.shape if img is not None else 'None'}, actions: {actions}")
            parquet_rows.append(into_parquet_row((img, actions)))

    # Convert to pyarrow Table and write to Parquet
    table = pa.Table.from_pylist(parquet_rows)
    os.makedirs(save_dir, exist_ok=True)
    pq.write_table(table, f'{save_dir}/{session_id}.parquet')
    print(f"Finished processing session {session_id}. Total rows: {len(parquet_rows)}")
    return parquet_rows

def into_parquet_row(pair):
    img, actions = pair
    mouse_action, mouse_x, mouse_y, scroll_action, keyboard_action = actions
    return {
        'img': img.tobytes() if img is not None else None,
        'shape': img.shape,
        'mouse_action': mouse_action,
        'mouse_x': mouse_x,
        'mouse_y': mouse_y,
        'scroll_action': list(scroll_action) if scroll_action is not None else None,
        'keyboard_action': list(keyboard_action) if keyboard_action is not None else None
    }

def test_read_parquet():
    file_path = '/Users/minjunes/vlm/scripts/6a0a0e1b-8226-4fa7-a4be-42207038b415.parquet'
    # Read the Parquet file
    table = pq.read_table(file_path)
    
    # Convert to pandas DataFrame for easier manipulation
    df = table.to_pandas()
    
    # Process each row
    for index, row in df.iterrows():
        # Convert frame back to numpy array
        if row['img'] is not None:
            shape = row['shape']
            frame = np.frombuffer(row['img'], dtype=np.uint8).reshape((shape[1], shape[0], shape[2]))
        else:
            frame = None
        
        # Convert scroll_action and keyboard_action back to tuples if they exist
        scroll_action = tuple(row['scroll_action']) if row['scroll_action'] is not None else None
        keyboard_action = tuple(row['keyboard_action']) if row['keyboard_action'] is not None else None
        
        # Create a DataRow object
        data_row = DataRow(
            frame=frame,
            mouse_action=row['mouse_action'],
            mouse_x=row['mouse_x'],
            mouse_y=row['mouse_y'],
            scroll_action=scroll_action,
            keyboard_action=keyboard_action
        )
        
        print(data_row)

if __name__ == '__main__':
    conn = connect_to_postgres()
    session_ids = get_unique_session_ids(conn)
    for session_id in tqdm(session_ids):
        rows = get_all_session_rows(conn, session_id)
        process_session(session_id, rows)
    conn.close()
