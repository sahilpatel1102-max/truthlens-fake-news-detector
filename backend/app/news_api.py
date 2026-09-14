import os
import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

API_URL = "https://newsapi.org/v2/everything"


def search_news(query, page_size=5):
    """
    Search NewsAPI for articles related to a query.
    """

    if not NEWS_API_KEY:
        raise ValueError(
            "NEWS_API_KEY is not configured."
        )

    params = {
        "q": query,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    return data.get("articles", [])


if __name__ == "__main__":

    query = input(
        "Enter a news topic to search: "
    )

    try:

        articles = search_news(query)

        if not articles:

            print("\nNo news articles found.")

        else:

            print(
                f"\nFound {len(articles)} article(s):\n"
            )

            for i, article in enumerate(
                articles,
                start=1
            ):

                print("=" * 60)

                print(f"Article #{i}")

                print(
                    "Title:",
                    article.get(
                        "title",
                        "N/A"
                    )
                )

                print(
                    "Source:",
                    article.get(
                        "source",
                        {}
                    ).get(
                        "name",
                        "N/A"
                    )
                )

                print(
                    "Published:",
                    article.get(
                        "publishedAt",
                        "N/A"
                    )
                )

                print(
                    "URL:",
                    article.get(
                        "url",
                        "N/A"
                    )
                )

    except Exception as e:

        print("\nError:")
        print(e)