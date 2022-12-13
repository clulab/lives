import pandas as pd
import wave
import os
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


######################################## diarisation #################################################################

turn_diarisation_en_df = pd.read_csv("./data_en/turn_diarisation.csv")
#print(turn_diarisation_en_df.head())
turn_diarisation_es_df = pd.read_csv("./data_es/turn_diarisation.csv")
print(turn_diarisation_es_df.iloc[0])
manual_diarisation_record_df = pd.read_csv("./data/manual_diarisation_record.csv")
#print(manual_diarisation_record_df.head())
recording_id = list(manual_diarisation_record_df["recording_id"])
print(recording_id)
print(type(recording_id))

# Rename the audio file's name in the "wav_file_name" feature to match the audio file's name
# in the "manual_diarisation_record.csv"
for i in range(len(turn_diarisation_es_df["wav_file_name"])):
    for filename in recording_id:
        if turn_diarisation_es_df["wav_file_name"][i].startswith(filename):
            turn_diarisation_es_df["wav_file_name"][i] = filename + ".wav"
print(turn_diarisation_es_df)

# Integrate the datasets together, and called "turn_diarisation".
turn_diarisation = pd.concat([turn_diarisation_en_df, turn_diarisation_es_df],ignore_index=True)
print('diarisation:\n', turn_diarisation.iloc[0])

# Calculate the length of each turn.
turn_diarisation["turn_length"] = None
turn_diarisation["language"] = None
print(manual_diarisation_record_df.iloc[0])
for i in range(len(turn_diarisation)):
    turn_diarisation['turn_length'][i] = (turn_diarisation['turn_end'][i] - turn_diarisation['turn_start'][i])
    name, extension = os.path.splitext(turn_diarisation['wav_file_name'][i])
    for j in range(len(manual_diarisation_record_df)):
        if name == manual_diarisation_record_df['recording_id'][j]:
            turn_diarisation['language'][i] = manual_diarisation_record_df['interviw_language'][j]
            turn_diarisation['language'][i] = turn_diarisation['language'][i].lower()

print(turn_diarisation.iloc[0])

##### Split the audio by turn length

# Get the path_audio, the audio's name with the file format, and the audio's name.
audio_name_path = []
file = []
file_name = []
for filename in os.listdir("./data"):
    if filename.endswith(".wav"):
        audio_name_path.append(os.path.join("./data", filename))
        pathname, extension = os.path.splitext(filename)
        file.append(filename)
        file_name.append(pathname)
print(audio_name_path)
print(file)
print('file_name:\n', file_name)

# Read the audio files and extract information from the audio files.
result = []
for name in range(len(audio_name_path)):
    with wave.open(audio_name_path[name], "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        segment = []
        for i in range(len(turn_diarisation)):
            # Set position in wave to start of segment:
            #print(turn_diarisation['turn_start'][i] * framerate)
            #print(int(turn_diarisation['turn_start'][i] * framerate))
            if turn_diarisation['wav_file_name'][i] == file[name]:
                infile.setpos(int(turn_diarisation['turn_start'][i] * framerate))
                #print(int(turn_diarisation['turn_start'][i] * framerate))
                # Extract data
                data = infile.readframes(int(turn_diarisation['turn_length'][i] * framerate))
                #print('data', data)
                segment.append(data)
    result.append(segment)
print(len(result)) # Check there are 8 audio files.
#print(result)

# Write the extracted data into small chunks of audio files.
for i in range(len(result)):
    # Check each interview has how many turns
    print(len(result[i]))
    for idx in range(len(result[i])):
        with wave.open('./split_audio/' + file_name[i] + "_" + str(idx+1) + ".wav", "w") as outfile:
            outfile.setnchannels(nchannels)
            outfile.setsampwidth(sampwidth)
            outfile.setframerate(framerate)
            outfile.setnframes(int(len(result[i][idx]) / sampwidth))
            #print(len(result[i][idx]))
            outfile.writeframes(result[i][idx])

turn_diarisation["split_wav_file_name"] = None
turn_diarisation['count'] = None
print(turn_diarisation.iloc[0])

# Count for each interview
for i in range(len(file)):
    print(file[i])
    count = 0
    for j in range(len(turn_diarisation)):
        if turn_diarisation["wav_file_name"][j] == file[i]:
            count +=1
            turn_diarisation['count'][j] = count

# Assign values to the "split_wav_file_name" column.
for i in range(len(turn_diarisation["wav_file_name"])):
    name, extension= os.path.splitext(turn_diarisation["wav_file_name"][i])
    #print(name)
    turn_diarisation["split_wav_file_name"][i] = (name + "_" + str(turn_diarisation["count"][i]) + extension)

# Drop the "count" column.
turn_diarisation.drop('count', axis=1, inplace=True)
print(turn_diarisation.iloc[0])

## Run the split_audio_transcribe.py and split_audio_translate.py to get the transcriptions and translations.
# Then, integrate the transcriptions and translations to the "turn_diarisation" dataset.
split_audio_transcription = pd.read_csv('./output/split_audio_transcription.csv')
print('split_transcription:\n', split_audio_transcription)
split_audio_translation = pd.read_csv('./output/split_audio_translation.csv')
split_audio_translation.drop('language', axis=1, inplace=True)
split_audio_translation = split_audio_translation.rename(columns={'transcription':'translation',
                                                                  'wav_file_name':'split_wav_file_name'})
print('split_translation:\n', split_audio_translation)
split_audio_transcription_translation = pd.merge(split_audio_transcription, split_audio_translation, how="left",
                                                 on=['split_wav_file_name'])

print(len(split_audio_transcription_translation))

split_audio_transcription_translation['transcribe_time_length'] = round(split_audio_transcription_translation['transcribe_time_length'], 2)
split_audio_transcription_translation['translate_time_length'] = round(split_audio_transcription_translation['translate_time_length'], 2)
print(split_audio_transcription_translation.iloc[0])
split_audio_transcription_translation.to_csv('./data/split_audio_transcription_translation.csv', index=False)

# Integrate the turn_diarisation dataset and split_audio_transcription_translation dataset.
split_audio_transcription_translation.drop(['language', 'transcribe_time_length', 'translate_time_length'],
                                           axis=1, inplace=True)
turn_diarisation = pd.merge(turn_diarisation, split_audio_transcription_translation, how='left',
                            on=['split_wav_file_name'])

# For the Spanish audio files, replace the transcriptions with translations.
for i in range(len(turn_diarisation)):
    if turn_diarisation['language'][i] == "spanish":
        turn_diarisation['transcription'][i] = turn_diarisation['translation'][i]

# Drop the translation feature
turn_diarisation.drop('translation', axis=1, inplace=True)
turn_diarisation.to_csv('./data/test.csv', index=False)

# Split the turn_diarisation dataset into two datasets: coach and participant.
## the turn_diarisation_coach
turn_diarisation_coach = turn_diarisation.loc[turn_diarisation.speaker_type == 'coach']
turn_diarisation_coach = turn_diarisation_coach.reset_index(drop=True)
print(turn_diarisation_coach)
turn_diarisation_coach.to_csv("./data/turn_diarisation_coach.csv", index=False)

## the turn_diariation_participant
turn_diarisation_participant = turn_diarisation.loc[turn_diarisation.speaker_type == 'participant']
turn_diarisation_participant = turn_diarisation_participant.reset_index(drop=True)
print(turn_diarisation_participant)
turn_diarisation_participant.to_csv("./data/turn_diarisation_participant.csv", index=False)


######################################## annotation ################################################################

# Integrate the turn_annotation datasets.
turn_annotation_en = pd.read_csv("./data_en/turn_annotation.csv")
turn_annotation_es = pd.read_csv("./data_es/turn_annotation.csv")
turn_annotation = pd.concat([turn_annotation_en, turn_annotation_es], ignore_index=True)

# Rename the audio file's name in the "wav_file_name" feature to match the audio file's name
# in the "manual_diarisation_record.csv"
for i in range(len(turn_annotation["wav_file_name"])):
    for filename in recording_id:
        if turn_annotation["wav_file_name"][i].startswith(filename):
            turn_annotation["wav_file_name"][i] = filename + ".wav"
print(turn_annotation)
turn_annotation.to_csv("./data/turn_annotation.csv", index=False)

turn_annotation_choices = turn_annotation.loc[turn_annotation.variable_type == 'choices']
print(len(turn_annotation_choices))
turn_annotation_choices.to_csv("./data/turn_annotation_choices.csv", index=False)

# The annotations for the coach
# The turn_annotation_choices_coach dataset is for visualizing the distribution of annotations
turn_annotation_choices_coach = turn_annotation_choices[turn_annotation_choices["variable_subtype"].str.startswith('coach')]
turn_annotation_choices_coach.to_csv("./data/turn_annotation_choices_coach.csv", index=False)

# Based on the distribution of annotations, select one annotation from categories.
turn_annotation_mia_coach = turn_annotation.loc[turn_annotation.annotation == "mi_adherent_mia"]
turn_annotation_mia_coach = turn_annotation_mia_coach.reset_index(drop=True)
print('\n\nmia:\n\n',turn_annotation_mia_coach)
turn_annotation_mia_coach.to_csv('./data/turn_annotation_mia_coach.csv', index=False)

# The annotations for the participant
# The annotation_annotation_choices_participant is for visualizing the distribution of annotations
turn_annotation_choices_participant = turn_annotation_choices[turn_annotation_choices["variable_subtype"].str.startswith('participant')]
turn_annotation_choices_participant.to_csv("./data/turn_annotation_choices_participant.csv", index=False)

# Based on the distribution of annotations, select one annotation from categories.
turn_annotation_change_talk_participant = turn_annotation.loc[turn_annotation.annotation ==
                                                              "change_talk_goal_talk_and_opportunities"]
turn_annotation_change_talk_participant = turn_annotation_change_talk_participant.reset_index(drop=True)
print('change talk:\n\n',turn_annotation_change_talk_participant)
turn_annotation_change_talk_participant.to_csv('./data/turn_annotation_change_talk_participant.csv', index=False)


################# Integrate the turn diarisation dataset and the turn annotation dataset ##########################
# Create two new columns in turn_diarisation dataframe: "variable_subtype" and "annotation"

# the dataset for coach
turn_diarisation_coach = pd.merge(turn_diarisation_coach, turn_annotation_mia_coach,
                                  how='left', on=['task_id', 'interview_id', 'annotator_id', 'wav_file_name',
                                                  'annotation_id'])
turn_diarisation_coach.to_csv('./data/turn_diarisation_annotation_coach.csv', index=False)

# the dataset for participant
turn_diarisation_participant = pd.merge(turn_diarisation_participant, turn_annotation_change_talk_participant,
                                        how='left', on=['task_id', 'interview_id', 'annotator_id', 'wav_file_name',
                                                        'annotation_id'])
turn_diarisation_participant.to_csv("./data/turn_diarisation_annotation_participant.csv", index=False)


### Reduce demiensions of the dataset and prepare the dataset for classifier.
stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# Coach
print(turn_diarisation_coach.iloc[0])
turn_diarisation_annotation_coach = turn_diarisation_coach[['transcription', 'annotation']]
turn_diarisation_annotation_coach['label'] = None
for i in range(len(turn_diarisation_annotation_coach)):
    if turn_diarisation_annotation_coach['annotation'][i] == "mi_adherent_mia":
        turn_diarisation_annotation_coach['label'][i] = bool(True)
    else:
        turn_diarisation_annotation_coach['label'][i] = bool(False)

print(turn_diarisation_annotation_coach.iloc[0])
# Check if there is any null in the transcription
turn_diarisation_annotation_coach = turn_diarisation_annotation_coach.loc[turn_diarisation_annotation_coach.transcription.notnull()]
turn_diarisation_annotation_coach = turn_diarisation_annotation_coach.reset_index(drop=True)

# Create a new column called "transcription_NoStopWords".
turn_diarisation_annotation_coach['transcription_NoStopWords'] = turn_diarisation_annotation_coach['transcription'].apply(
    lambda  x : " ".join([i for i in re.sub("[^a-zA-Z]", " ", x).lower().split() if i not in stopwords]))

# Create a new column called "transcription_afterLemmatization"
turn_diarisation_annotation_coach['transcription_afterLemmatization'] = turn_diarisation_annotation_coach['transcription'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", x).lower().split()]))

# Create a new column called "transcription_After_RemovingStopWords_lemma"
turn_diarisation_annotation_coach['transcription_After_RemovingStopWords_lemma'] = turn_diarisation_annotation_coach['transcription_NoStopWords'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", x).lower().split()]))

turn_diarisation_annotation_coach.to_csv('./data/turn_coach_for_model.csv', index=False)

# Participant
turn_diarisation_annotation_participant = turn_diarisation_participant[['transcription', 'annotation']]
turn_diarisation_annotation_participant['label'] = None
for i in range(len(turn_diarisation_annotation_participant)):
    if turn_diarisation_annotation_participant['annotation'][i] == "change_talk_goal_talk_and_opportunities":
        turn_diarisation_annotation_participant['label'][i] = bool(True)
    else:
        turn_diarisation_annotation_participant['label'][i] = bool(False)

print(turn_diarisation_participant.iloc[0])
turn_diarisation_annotation_participant = turn_diarisation_annotation_participant.loc[turn_diarisation_annotation_participant.transcription.notnull()]
turn_diarisation_annotation_participant = turn_diarisation_annotation_participant.reset_index(drop=True)

# Create a new column called "transcription_NoStopWords".
turn_diarisation_annotation_participant['transcription_NoStopWords'] = turn_diarisation_annotation_participant['transcription'].apply(
    lambda x : " ".join([i for i in re.sub("[^a-zA-Z]", " ", x).lower().split() if i not in stopwords]))

# Create a new column called "transcription_afterLemmatization"
turn_diarisation_annotation_participant['transcription_afterLemmatization'] = turn_diarisation_annotation_participant['transcription'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", x).lower().split()]))

# Create a new column called "transcription_After_RemovingStopWords_lemma"
turn_diarisation_annotation_participant['transcription_After_RemovingStopWords_lemma'] = turn_diarisation_annotation_participant['transcription_NoStopWords'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " ", x).lower().split()]))
print(turn_diarisation_annotation_participant.info())
turn_diarisation_annotation_participant.to_csv('./data/turn_participant_for_model.csv', index=False)