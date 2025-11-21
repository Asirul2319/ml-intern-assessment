from ngram_model import TrigramModel

def main():
    # Create a new TrigramModel
    model = TrigramModel()

    # Train the model on the example corpus
    # with open("data/example_corpus.txt", "r") as f:
    with open("data/Alice_Adventures_in_Wonderland_by_Lewis_Carroll.txt", "r", encoding="utf-8") as f:
        text = f.read()
    model.fit(text)

    # Generate new text
    generated_text = model.generate()
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
