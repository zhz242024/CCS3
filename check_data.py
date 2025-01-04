#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("prepare_gaze_pupil.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    root_dir = "/home/csd440/slurm_test/data/participants_data" 

    all_files = os.listdir(root_dir)
    logger.info(f"Found {len(all_files)} files in {root_dir}")

    pattern_gaze = re.compile(r"^(Participant\d+)_Gaze\.xlsx$", re.IGNORECASE)
    pattern_pupil = re.compile(r"^(Participant\d+)_Pupil\.xlsx$", re.IGNORECASE)

    participant_dict = {}

    for f in all_files:
        gaze_match = pattern_gaze.match(f)
        pupil_match = pattern_pupil.match(f)

        if gaze_match:
            pid = gaze_match.group(1)
            if pid not in participant_dict:
                participant_dict[pid] = {}
            participant_dict[pid]["gaze"] = f
        elif pupil_match:
            pid = pupil_match.group(1)
            if pid not in participant_dict:
                participant_dict[pid] = {}
            participant_dict[pid]["pupil"] = f

    logger.info(f"Identified {len(participant_dict)} participants.")

    for pid, file_info in tqdm(participant_dict.items(), desc="Processing Participants"):
        gaze_file = file_info.get("gaze", None)
        pupil_file = file_info.get("pupil", None)

        if gaze_file:
            gaze_path = os.path.join(root_dir, gaze_file)
            try:
                gaze_dict = pd.read_excel(gaze_path, sheet_name=None)
                for sheet_name, df in gaze_dict.items():
                    logger.info(f"{pid}, Gaze, {sheet_name}: {df.shape[0]} rows")
            except Exception as e:
                logger.error(f"Failed to read Gaze file {gaze_path}: {e}")

        if pupil_file:
            pupil_path = os.path.join(root_dir, pupil_file)
            try:
                pupil_dict = pd.read_excel(pupil_path, sheet_name=None)
                for sheet_name, df in pupil_dict.items():
                    logger.info(f"{pid}, Pupil, {sheet_name}: {df.shape[0]} rows")
            except Exception as e:
                logger.error(f"Failed to read Pupil file {pupil_path}: {e}")

if __name__ == "__main__":
    main()

