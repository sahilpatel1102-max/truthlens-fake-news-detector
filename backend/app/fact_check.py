import os
import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def search_fact_checks(query):
    """
    Search Google Fact Check for previously fact-checked claims.
    """

    if not API_KEY:
        raise ValueError(
            "GOOGLE_FACT_CHECK_API_KEY is not configured."
        )

    params = {
        "query": query,
        "key": API_KEY,
        "pageSize": 10
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    return data.get("claims", [])


if __name__ == "__main__":

    query = input(
        "Enter a claim to fact-check: "
    )

    try:

        claims = search_fact_checks(query)

        if not claims:
            print("\nNo fact-check results found.")
        else:

            print(
                f"\nFound {len(claims)} fact-check result(s):\n"
            )

            for i, claim in enumerate(claims, start=1):

                print("=" * 60)

                print(f"Result #{i}")

                print(
                    "Claim:",
                    claim.get("text", "N/A")
                )

                reviews = claim.get(
                    "claimReview",
                    []
                )

                for review in reviews:

                    publisher = review.get(
                        "publisher",
                        {}
                    )

                    print(
                        "Publisher:",
                        publisher.get(
                            "name",
                            "N/A"
                        )
                    )

                    print(
                        "Rating:",
                        review.get(
                            "textualRating",
                            "N/A"
                        )
                    )

                    print(
                        "URL:",
                        review.get(
                            "url",
                            "N/A"
                        )
                    )

    except Exception as e:

        print("\nError:")
        print(e)