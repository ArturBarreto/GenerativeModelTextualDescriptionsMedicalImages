import os
from os import makedirs
from os.path import dirname
from itertools import chain
from collections import Counter
from shutil import copy2
from datetime import datetime
from struct import unpack
import random
from PIL import Image

random.seed(111)


def get_files(dictionary: {}, values_of_interest: [], operation="AND"):

    if operation != "AND" and operation != "OR":
        return "BAD OPERATION!"

    returned_files = set()

    for key, values in dictionary.items():
        add_file = True
        for value_of_interest in values_of_interest:
            if value_of_interest in values["keywords"] and add_file:
                if operation == "OR":
                    returned_files.add(key)
            else:
                if operation == "AND":
                    add_file = False
                    break
        if operation == "AND" and add_file:
            returned_files.add(key)

    return returned_files

class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while True:
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]
            if len(data) == 0:
                break


RADIOLOGY = True

if not RADIOLOGY:
    non_radiology = "non-"
else:
    non_radiology = ""

keywords_file_path_test = "C:/Users/artur/OneDrive/Mestrado/roco-dataset/data/test/" + non_radiology + "radiology/keywords.txt"
keywords_file_path_train = "C:/Users/artur/OneDrive/Mestrado/roco-dataset/data/train/" + non_radiology + "radiology/keywords.txt"
keywords_file_path_validation = "C:/Users/artur/OneDrive/Mestrado/roco-dataset/data/validation/" + non_radiology + "radiology/keywords.txt"

captions_file_path_test = "C:/Users/artur/OneDrive/Mestrado/roco-dataset/data/test/" + non_radiology + "radiology/captions.txt"
captions_file_path_train = "C:/Users/artur/OneDrive/Mestrado/roco-dataset/data/train/" + non_radiology + "radiology/captions.txt"
captions_file_path_validation = "C:/Users/artur/OneDrive/Mestrado/roco-dataset/data/validation/" + non_radiology + "radiology/captions.txt"

keywords_files_paths = [keywords_file_path_test, keywords_file_path_train, keywords_file_path_validation]
captions_files_paths = [captions_file_path_test, captions_file_path_train, captions_file_path_validation]

files_paths = zip(keywords_files_paths, captions_files_paths)

# print(list(files_paths))

MAX_NUM_CAPTION_TOKENS = -1

words = Counter()
list_words = []
dictionary = {}
flat_list = []

for keywords_file_path, captions_file_path in files_paths:
    list_words = []
    with open(file=keywords_file_path, mode="r", encoding="utf-8") as keywords_file, \
         open(file=captions_file_path, mode="r", encoding="utf-8") as captions_file:

        type_path = keywords_file_path.split("/")[7]
        # print(type_path)

        for keywords_line, captions_line in zip(keywords_file, captions_file):
            keywords_line = keywords_line.replace("\t", " ")
            key = keywords_line.split()[0]
            keywords = keywords_line.split()[1:]
            captions = captions_line.split()[1:]

            # dictionary.update({key: []})
            dictionary[key] = {"type_path": type_path, "keywords": keywords, "captions": captions}

            list_words.append(keywords)

        flat_list = list(chain(*list_words))
        words.update(flat_list)

# print(len(dictionary.keys()))
# print(dictionary)
# print(list_words)
# print(flat_list)
print("Num of words: " + str(len(words.most_common())))
print(words.most_common())

types_exam = ['ct', 'scan', 'tomography', 'radiograph', 'xray', 'mri', 'contrast', 'resonance', 'magnetic', 'abdominal',
              'ultrasound', 'angiography', 'angiogram', 'catheter']
bodys_part = ['chest', 'artery', 'abdoman', 'lobe', 'lung', 'bone', 'tissue', 'month', 'pulmonary', 'head', 'brain',
              'vein', 'liver', 'ventricle', 'pelvi', 'kidney', 'spine', 'neck', 'pleural', 'muscle', 'renal',
              'coronary', 'femoral', 'cervical', 'atrium', 'bowel', 'aorta', 'aortic', 'hip', 'heart', 'tooth']
problems = ['tumor', 'fracture', 'normal', 'cystic', 'cyst', 'effusion', 'calcification', 'nodule', 'hepatic', 'node',
            'stent', 'heterogeneou', 'pancreatic', 'week', 'aneurysm', 'edema', 'irregular', 'dilatation', 'absces',
            'disease', 'apical', 'hematoma', 'fistula', 'cancer']

all_values_of_interest = []
for type in types_exam:
    for body in bodys_part:
        for problem in problems:
            all_values_of_interest.append([type, body, problem])

# print(all_values_of_interest)

destination = "C:/Users/artur/OneDrive/Mestrado/roco-dataset/output/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"
makedirs(dirname(destination), exist_ok=True)

# all_values_of_interest = []
# all_values_of_interest.append(['angiogram', 'artery', 'aneurysm'])
# all_values_of_interest.append(['ct', 'head', 'pancreatic'])
# all_values_of_interest.append(['tomography', 'lung', 'nodule'])
# all_values_of_interest.append(['xray', 'chest', 'normal'])
# all_values_of_interest.append(['radiograph', 'hip', 'fracture'])
# all_values_of_interest.append(['scan', 'abdoman', 'cystic'])

total_files = 0
for values_of_interest in all_values_of_interest:

    files = sorted(get_files(dictionary, values_of_interest, "AND"))
    # print(sorted(files))

    if 40 <= len(files) <= 80:

        # destination = "C:/Users/artur/OneDrive/Mestrado/roco-dataset/output/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-'.join(values_of_interest) + "/"
        # makedirs(dirname(destination), exist_ok=True)

        quant_files = 0
        for file in files:
            # print(file)
            # print(dictionary.get(file))

            source = "C:/Users/artur/OneDrive/Mestrado/roco-dataset/data/" \
                     + dictionary.get(file)["type_path"] + "/" + non_radiology + "radiology/images/" + file + ".jpg"
            # print(source)

            try:
                image = JPEG(source)
                image.decode()

                path = destination + file + ".jpg"

                if not os.path.isfile(path):
                    quant_files = quant_files + 1
                    copy2(source, destination)

                    basewidth = 500
                    img = Image.open(path)
                    if img.size[0] > img.size[1]:
                        wpercent = (basewidth / float(img.size[0]))
                        hsize = int((float(img.size[1]) * float(wpercent)))
                        img = img.resize((basewidth, hsize), Image.LANCZOS)
                    else:
                        hpercent = (basewidth / float(img.size[1]))
                        wsize = int((float(img.size[0]) * float(hpercent)))
                        img = img.resize((wsize, basewidth), Image.LANCZOS)

                    img.save(path)

                    with open(file=destination + "1_keywords.txt", mode="a", encoding="utf-8") as keywords_file, \
                         open(file=destination + "2_captions.txt", mode="a", encoding="utf-8") as captions_file, \
                         open(file=destination + "3_valofint.txt", mode="a", encoding="utf-8") as valofint_file:

                        # for value in values_of_interest:
                        #     dictionary.get(file)["keywords"].remove(value)

                        for i in range(1):
                            keywords_file.write(file + ".jpg" + "#" + str(i) + "\t")
                            keywords_file.write(" ".join(dictionary.get(file)["keywords"]))
                            keywords_file.write("\n")
                            # random.shuffle(dictionary.get(file)["keywords"])

                            captions_file.write(file + ".jpg" + "#" + str(i) + "\t")
                            captions_file.write(" ".join(dictionary.get(file)["captions"]))
                            captions_file.write("\n")
                            # random.shuffle(dictionary.get(file)["captions"])

                            valofint_file.write(file + ".jpg" + "#" + str(i) + "\t")
                            valofint_file.write(" ".join(values_of_interest))
                            valofint_file.write("\n")

            except:
                print("Não foi possível copiar o arquivo: " + source)

            # print(dictionary.get(file)["keywords"])
            # print(dictionary.get(file)["captions"])

        print(" ".join(values_of_interest) + ": " + str(quant_files))

        total_files = total_files + quant_files

        with open(file=destination + "0_values_of_interest.txt", mode="a", encoding="utf-8") as values_of_interest_file:
            values_of_interest_file.write(" ".join(values_of_interest) + " " + str(quant_files) + "\n")

with open(file=destination + "0_values_of_interest.txt", mode="a", encoding="utf-8") as values_of_interest_file:
    values_of_interest_file.write("Total = " + str(total_files) + "\n")

print("Total = " + str(total_files) + "\n")
