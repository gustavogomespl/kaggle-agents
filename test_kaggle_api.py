"""Manual Kaggle API smoke test (never executed during pytest collection)."""


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    submissions = api.competition_submissions("playground-series-s6e2")
    if not submissions:
        print("No submissions found")
        return

    submission = submissions[0]
    print(dir(submission))
    print("publicScore:", getattr(submission, "publicScore", None))
    print("status:", getattr(submission, "status", None))
    print("errorDescription:", getattr(submission, "errorDescription", None))


if __name__ == "__main__":
    main()
