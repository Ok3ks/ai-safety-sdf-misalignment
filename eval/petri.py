from typing import List
from inspect_ai import eval
from inspect_ai.model import get_model
from inspect_petri import audit

def main(target:str, auditor: str, judge:str):
    """Evaluates a target model variant across seeds and dimensions"""
    eval(
        audit(
            seed_instructions="tags:alignment_faking,power_seeking",
            max_turns=30, 
            realism_filter=True, 
            ),
        model_roles = dict(
            auditor=get_model(auditor),
            target=get_model(target),
            judge=get_model(judge),
            realism=get_model("anthropic/claude-haiku-4-5")
        )
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--auditor", choices=["anthropic/claude-opus-4-7", "anthropic/claude-opus-4-6"])
    parser.add_argument("-t", "--target")
    parser.add_argument("-j", "--judge", default="anthropic/claude-opus-4-7")

    args = parser.parse_args()
    main(args.target, args.auditor, args.judge)
