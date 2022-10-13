# calculate diarization error rate using pyannote

from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
from pyannote.core import Annotation
from pyannote.audio.utils import metric
from pyannote.database.util import load_rttm
#from pyannote.audio
import numpy as np
from pathlib import Path
import pandas as pd


def calculate_der(gold_rttm_path, pred_rttm_path):
    """
    Use the gold rttm and predicted rttm to calculate DER
    """
    # load gold rttm
    gold_annotation = load_this_rttm(gold_rttm_path)
    #print(gold_annotation)

    # load predicted rttm
    pred_annotation = load_this_rttm(pred_rttm_path)
    #print(pred_annotation)

    # calculate DER
    metric = DiarizationErrorRate()
    der = metric(gold_annotation, pred_annotation, detailed=True)

    return der


def return_errors_as_rttm(gold_rttm_path, pred_rttm_path):
    # load rttms
    gold_annotation = load_this_rttm(gold_rttm_path)
    pred_annotation = load_this_rttm(pred_rttm_path)

    # get difference
    diff = IdentificationErrorAnalysis().difference(reference=gold_annotation,
                                                  hypothesis=pred_annotation)

    save_name = pred_rttm_path.split(".rttm")[0] + "_difference.rttm"

    with open(save_name, 'w') as save_f:
        diff.write_rttm(save_f)


def return_missed_detection_as_json(gold_rttm_path, pred_rttm_path):
    """
    Return missed detection errors in json format
    Needs to be altered from rttm format due to
    the extra fields in 'difference' Annotations
    """
    gold_annotation = load_this_rttm(gold_rttm_path)
    pred_annotation = load_this_rttm(pred_rttm_path)

    # get difference
    diff = IdentificationErrorAnalysis().difference(reference=gold_annotation,
                                                    hypothesis=pred_annotation)

    # convert to nested dict format for reading into json
    diff_json = diff.for_json()
    diff_json = [item for item in diff_json['content'] if item['label'][0] == 'missed detection']

    # reformat for ease of interpretation
    for item in diff_json:
        # put start and end in their own categories
        item['start'] = item['segment']['start']
        item['end'] = item['segment']['end']
        item['length'] = item['end'] - item['start']
        del item['segment']
        # put missed detection and speaker in their own categories
        item['speaker'] = item['label'][1]
        item['label'] = item['label'][0]
        # delete track since this is now redundant
        del item['track']

    # convert to pandas df
    diff_df = pd.DataFrame(diff_json)

    # change name for saving
    save_name = pred_rttm_path.split(".rttm")[0] + "_missed.csv"

    # save
    diff_df.to_csv(save_name, index=False)


def calc_global_der(gold_rttm_basepath, pred_rttm_basepath, matching_filenames, skip_overlap=False):
    """
    Calculate the global DER across files
    Expects gold and preds files to have matching names
    overlapping: whether to include overlapping speech in calculations
    """
    all_results = []
    for f in matching_filenames:
        gold_annotation = load_this_rttm(f"{gold_rttm_basepath}/{f}")
        pred_annotation = load_this_rttm(f"{pred_rttm_basepath}/{f}")

        all_results.append((gold_annotation, pred_annotation))

    # calculate DER
    metric = DiarizationErrorRate(skip_overlap=skip_overlap)

    for (gold, pred) in all_results:
        metric(gold, pred)

    global_value = abs(metric)
    mean, (lower, upper) = metric.confidence_interval()

    return mean, lower, upper


def load_this_rttm(rttm_path):
    """
    Load an RTTM file that should have ONE k, v pair
    """
    # load gold rttm
    rttm_dict = load_rttm(rttm_path)
    # this is a dict and should have a single key
    rkey = list(rttm_dict.keys())
    if len(rkey) != 1:
        exit("This dict doesn't have the right number of annotations"
             "Annotations expected: 1"
             f"Annotations seen: {len(rkey)}")
    else:
        the_annotation = rttm_dict[rkey[0]]

    return the_annotation


if __name__ == "__main__":
    gold_path = "/media/jculnan/backup/From LIVES folder/diarization_effort/manually_annotated/test_all"
    pred_path = "/media/jculnan/backup/From LIVES folder/diarized_files"

    gold_files = Path(gold_path).glob('**/*')
    golds = [f.name for f in gold_files if f.is_file()]

    ids = []
    for f in Path(pred_path).iterdir():
        if f.name in golds:
            ids.append(f.name)
            #mean_der, lower_der, upper_der = calc_global_der(gold_path, pred_path, f.name)

    #         print("Global scores WITH ov")
    #
    # ids = {#"RE03d2270a7c67f2b3bb55edf7dfc7901a.rttm",
    #        #"RE06769651e62d06f4d6dfbe03efbbe024.rttm",
    #        #"RE1066da447ef158e74394647559d8540f.rttm",
    #        #"RE13969403f5a37da7308bac270bd49939.rttm",
    #        #"RE27f79e278003a6c6f61e2c8fe7004703.rttm",
    #        "RE2ae3a8526e6b62f71edfca298403456c.rttm",
    #        #"RE31cdaf16f092e76c9920f1d6fe6a4750.rttm",
    #        #"RE40520e19a322f0c4ac342626b73ec77f.rttm",
    #        #"RE4074882f3e5ff29f3ac9412a8215f0d8.rttm",
    #        #"RE48495771e7fac4225ec1ef76101a09e3.rttm"
    #        }

    mean_der, lower_der, upper_der = calc_global_der(gold_path, pred_path, ids)

    print("Global scores WITH overlapping speech")
    print(mean_der)
    print(lower_der)
    print(upper_der)
    # exit()

    mean_der2, lower_der2, upper_der2 = calc_global_der(gold_path, pred_path, ids, skip_overlap=True)

    print("Global scores WITHOUT overlapping speech")
    print(mean_der2)
    print(lower_der2)
    print(upper_der2)

    with open("/media/jculnan/backup/From LIVES folder/diarization_effort/pyannote_der_info.csv", 'w') as f:
        f.write("id,total,confusion,correct,false_alarm,missed_detection,der\n")

        for id in ids:
            this_gold = f"{gold_path}/{id}"
            this_pred = f"{pred_path}/{id}"

            # return_missed_detection_as_json(this_gold, this_pred)

            der = calculate_der(this_gold, this_pred)
            f.write(f"{id},{der['total']},{der['confusion']},"
                    f"{der['correct']},{der['false alarm']},"
                    f"{der['missed detection']},"
                    f"{der['diarization error rate']}\n")
            print(f"id: {id} der: {der}")