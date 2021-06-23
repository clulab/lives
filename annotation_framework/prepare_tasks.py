import dataclasses
import json
import logging
import os
import re
import xml.etree.ElementTree as et

logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass(frozen=True)
class Utterance:
    start_time: float
    speaker: str
    words: list


tasks = []
for trs_file_name in os.listdir("transcript"):
    audio_file_name = re.sub("[.]trs$", ".wav", trs_file_name)
    trs_path = os.path.join("transcript", trs_file_name)
    audio_path = os.path.join("audio", audio_file_name)
    if not os.path.exists(audio_path):
        logging.warning(f"missing audio file {audio_path}")
        continue
    utterances = []
    for turn_elem in et.parse(trs_path).findall(".//Turn"):
        if not utterances or turn_elem.get("speaker") != utterances[-1].speaker:
            utterances.append(Utterance(
                start_time=float(turn_elem.get("startTime")),
                speaker=turn_elem.get("speaker"),
                words=[],
            ))
        utterances[-1].words.extend(turn_elem.itertext())
    tasks.append(dict(
        audio_file=audio_file_name,
        transcript_file=trs_file_name,
        audio=f'data/local-files/?d={audio_path}',
        text=''.join(f'{u.start_time:6.1f} {u.speaker}: {" ".join(u.words)}\n'
                     for u in utterances),
    ))


tasks_path = "tasks.json"
with open(tasks_path, 'w') as tasks_file:
    json.dump(tasks, tasks_file, indent=1)

logging.info(f"wrote {len(tasks)} task(s) to {tasks_path}")
