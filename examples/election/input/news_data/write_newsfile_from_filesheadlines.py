import json

if __name__ == "__main__":
    # create 3 news files from news_files with no bias, bias for bill, and bias for bradley
    root_name = "examples/election/news_data/v1_news"
    for adapted_headlines_filename in ["no_bias", "bill_bias", "bradley_bias"]:
        with open(root_name + "_headlines_" + adapted_headlines_filename + ".json") as f:
            adapted_headlines = json.load(f)
        with open(root_name + ".json") as f:
            original_news_dict = json.load(f)
        with open(root_name + "_" + adapted_headlines_filename + ".json", "w") as fp:
            json.dump(
                dict(zip(adapted_headlines, original_news_dict.values(), strict=False)),
                fp,
                indent=4,
            )
