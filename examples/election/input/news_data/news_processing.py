# NA prefetching and transforming the headlines
def fetch_and_transform_headlines(upload_file=True, file_dir="cached_headlines.json"):
    if upload_file:
        # os.chdir("src/election_sim")
        if file_dir is None:
            raise ValueError("Please provide a file directory")
        with open("src/election_sim/news_data" + file_dir) as f:
            headlines = json.load(f)
            return headlines
    else:  # generate headlines on fly
        api_key = "28b2e2855863475b99f771933d38f2f5"

        # Query parameters
        query = "environment sustainability climate"
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={query}&"
            "language=en&"
            "sortBy=publishedAt&"
            "pageSize=100&"
            f"apiKey={api_key}"
        )

        response = requests.get(url)
        data = response.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            raw_headlines = []
            for article in articles:
                title = article.get("title")
                if title != None:
                    # clean the  raw title
                    clean_title = title.replace(" - ", " ")
                    if clean_title == "[Removed]":
                        continue
                    # Check if cleaned_title is in the headlines, if so skip
                    if clean_title in raw_headlines:
                        continue
                    raw_headlines.append(clean_title)

            mapped_headlines, _ = transform_news_headline_for_sim(raw_headlines)

            return mapped_headlines
