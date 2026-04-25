import spacy, json, sys


def main():
    texts = json.loads(sys.stdin.read())
    nlp = spacy.load("en_core_web_sm", enable=["ner"])
    results = []
    for doc in nlp.pipe(texts, batch_size=1024, n_process=-1):
        results.append([(ent.text, ent.label_) for ent in doc.ents])
    print(json.dumps(results))


if __name__ == "__main__":
    main()