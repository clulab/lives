import datetime
import subprocess

def wav_times():
    """Use subprocess and `afinfo` to get times from a './data/' folder with .wav files

    return
    ------
    wav_seconds: float
        Sum of seconds of all interviews in folder
    n_wavs: int
        Number of processed wav files
    """
    grep_results = subprocess.run(f'for f in {directory}*.wav; do afinfo $f | grep "estimated duration" | grep -Eo "[0-9]+\.[0-9]+" ; done', shell=True, check=True, capture_output=True)
    time_strings = grep_results.stdout.decode().split('\n')

    wav_seconds = 0
    n_wavs = 0
    for i_time in time_strings:
        try:
            wav_seconds += float(i_time)
        except:
            continue
        else:
            n_wavs += 1

    return wav_seconds, n_wavs

def secs_to_hms(total_seconds:float):
    return str(datetime.timedelta(seconds=total_seconds))

if __name__ == __main__:
    directory = './data/'
    interview_seconds, n_ints = def wav_times()
    print(f"The total time is: {secs_to_hms(interview_seconds)}")
    print(f"The total time in seconds is: {interview_seconds}")
    print(f"The average interview duration is: {secs_to_hms(interview_seconds/n_ints)}")

