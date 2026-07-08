import pandas as pd


aspect = pd.read_csv("data/aspect_dictionary.csv")


aspect_dict = {}

for _, row in aspect.iterrows():
    aspect_dict[row["keyword"].lower()] = row["aspect"]



def predict(sentence):

    words = sentence.split()

    output = []

    i = 0

    while i < len(words):

        found = False


        if i + 1 < len(words):

            phrase = (
                words[i] + " " + words[i+1]
            ).lower()

            if phrase in aspect_dict:

                output.append({
                    "token": phrase,
                    "entity": "ASPECT",
                    "aspect": aspect_dict[phrase]
                })

                i += 2
                found = True


        if not found:

            word = words[i].lower()

            if word in aspect_dict:

                output.append({
                    "token": words[i],
                    "entity": "ASPECT",
                    "aspect": aspect_dict[word]
                })

            i += 1


    return output



hasil = predict(
    "Harga murah tapi packing jelek menggunakan si cepat"
)

print(hasil)