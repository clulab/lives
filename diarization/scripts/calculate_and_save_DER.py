# this is a script to calculate the DER for a set of files and
# save the output information to a CSV

import sys
# fixme
sys.path.append("/home/jculnan/github/lives")

from diarization.pyannote_der import calculate_der
from pathlib import Path


def get_fname_dict(fpath, nested=True):
    """
    Get the dict of full_filename : predicted rttm to use
    """
    names_dict = {}
    if nested:
        files = Path(fpath).glob("**/*")
    else:
        files = Path(fpath).iterdir()

    for f in files:
        if f.is_file():
            # if "combined" in f.name:
            #     names_dict[f.name] = f.name.split("_")[0] + ".rttm"
            names_dict[f.name] = f.name.split("_")[0] + ".rttm"

    return names_dict


def get_ids(predpath, goldlist):
    """
    Use a path to rttm prediction files and a list of gold file names
    To get all the files that should be compared
    """
    ids = []
    for f in Path(predpath).iterdir():
        if f.name in goldlist:
            ids.append(f.name)

    return ids


def get_subset_of_gold_dict(predpath, golddict):
    all_preds = [f.name for f in Path(predpath).iterdir()]

    return {k: v for k, v in golddict.items() if v in all_preds}


def get_der_and_save(goldpath, predpath, used_files, lang, save_name=None):
    """
    Get DER for list of files; save to CSV
    used_files: dict of full_filename : pred_filename
    """
    if save_name is None:
        save_name = "DER_output.csv"

    with open(save_name, 'a') as saver:
        for k, v in used_files.items():
            this_gold = f"{goldpath}/{k}"
            this_pred = f"{predpath}/{v}"

            der = calculate_der(this_gold, this_pred)
            saver.write(f"{k},{v},{lang},{der['diarization error rate']},{der['total']},"
                        f"{der['correct']},{der['missed detection']},"
                        f"{der['confusion']},{der['false alarm']}\n")


if __name__ == "__main__":
    save_name = None
    nested = True
    lang = "English"
    if len(sys.argv) > 1:
        gold_path = sys.argv[1]
        lang = gold_path.split("/")[-1]
        pred_path = sys.argv[2]
        if len(sys.argv) > 3:
            save_name = sys.argv[3]
            if len(sys.argv) > 4:
                nested = sys.argv[4]
    else:
        gold_path = "/media/jculnan/backup/From LIVES folder/diarization_effort/manually_annotated/test_spanish"
        pred_path = "/media/jculnan/backup/From LIVES folder/diarized_files"

    all_gold_names = get_fname_dict(gold_path, nested=nested)
    gold_sub = get_subset_of_gold_dict(pred_path, all_gold_names)
    #ids = get_ids(pred_path, all_gold_names)

    # todo: add save name
    #get_der_and_save(gold_path, pred_path, ids, save_name)
    get_der_and_save(gold_path, pred_path, gold_sub, lang, save_name)
