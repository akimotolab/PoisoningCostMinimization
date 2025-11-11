from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FeatureExtractor:
    def __init__(self, model_name="openai-community/gpt2"):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to("cuda:0")
        self.features = []

        def get_activation():
            def hook(model, input, output):
                self.features.append(input[0])

            return hook

        self.model.score.register_forward_hook(get_activation())

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_feature(self, raw_inputs):
        tokenized_inputs = self.tokenizer.encode(raw_inputs, return_tensors="pt").to("cuda:0")
        self.features.clear()
        self.model(tokenized_inputs)
        tokenized_inputs.to("cpu")

        return self.features
