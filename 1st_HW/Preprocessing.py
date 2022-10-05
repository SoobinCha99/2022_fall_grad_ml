import json
import codecs
import pprint
from sklearn.model_selection import train_test_split
import csv
import pandas as pd 
# You can import known libraries to write your own code


def load_json(file_name):
    """
    Load json file
    :param file_name: file name
    :return: loaded data from the file
    """
    with codecs.open(file_name, "r", "utf-8") as json_f:
        return json.load(json_f)

def extract_data(dataset):
    """
    Extract sentences and their labels from dataset
    - human_sentence: list of the first human utterance in a conversation
    - emotion_label: list of the emotion label.
    The range of the emotion label is 0 to 5
    0: 분노
    1: 슬픔
    2: 불안
    3: 상처
    4: 당황
    5: 기쁨
    :param dataset: loaded data from load_json function
    :return: human_sentences and their emotion labels
    """
    
    human_sentence = list()
    emotion_label = list()

    for i in range(0,len(dataset)): 
        human_sentence.append(dataset[i]['talk']['content']['HS01'])
        label = dataset[i]['profile']['emotion']['type']
        emotion_label.append(int([*label][1])-1)

    #pass    # Task 1

    return human_sentence, emotion_label


def save_csv(file_name, sentences, labels):
    """
    Save the sentences and labels to csv file
    Header is needed to identify the column

    :param file_name: output file name
    :param sentences: human sentences from extract_data function
    :param labels: emotion labels from extract_data function
    :return: None
    """
    temp_df = pd.DataFrame()
    temp_df["sentence"] = sentences
    temp_df["label"] = labels

    temp_df.to_csv(file_name,index=False)
    


def main():
    """
    Entry of the program
    :return: None
    """
    training_data_json_filename = "./data/train/감성대화말뭉치(최종데이터)_Training.json"
    test_data_json_filename = "./data/validataion/감성대화말뭉치(최종데이터)_Validation.json"

    training_data_csv_filename = "./processed_data/train.csv"
    valid_data_csv_filename = "./processed_data/valid.csv"
    test_data_csv_filename = "./processed_data/test.csv"

    test_size = 0.1
    random_state = 108

    # 1. Load json files
    training_data = load_json(training_data_json_filename)
    test_data = load_json(test_data_json_filename)
    #pprint.pprint(training_data[0])
    #print()

    # 2. Extract data
    training_sentences, training_labels = extract_data(training_data)
    test_sentences, test_labels = extract_data(test_data)

    # 3. Split training data to training and valid data
    # Task 3
    # Use train_test_split function
    # Use test_size and random_state values to split the dataset
    '''training_sentences = None
    valid_sentences = None
    training_labels = None
    valid_labels = None'''

    training_sentences, valid_sentences, training_labels, valid_labels = train_test_split(training_sentences, training_labels, test_size=test_size, random_state=random_state)

    print("# training data: ", len(training_sentences))
    print("# validation data: ", len(valid_sentences))
    print("# test data: ", len(test_sentences))

    # 4. Save data to csv file
    save_csv(training_data_csv_filename, training_sentences, training_labels)
    save_csv(valid_data_csv_filename, valid_sentences, valid_labels)
    save_csv(test_data_csv_filename, test_sentences, test_labels)

if __name__ == "__main__":
    main()
