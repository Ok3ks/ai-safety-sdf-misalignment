from huggingface_hub import snapshot_download


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo")
    parser.add_argument("-c", "--commit")

    args = parser.parse_args()
    
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.commit,
        # allow_patterns=["checkpoint-1000/*", "*adapter*"],
        token=True,
    )


