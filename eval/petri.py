from typing import List
from inspect_ai import eval
from inspect_ai.model import get_model
from inspect_petri import audit

def main(target:str, auditor: str, judge:str, judge_dimensions: List[int]):
    """Evaluates a target model variant across seeds and dimensions"""
    eval(
        audit(
            max_turns=30, 
            realism_filter=True, 
            judge_dimensions=judge_dimensions
            ),
        model_roles = dict(
            auditor=get_model(auditor, reasoning_effort="high"),
            target=get_model(target),
            judge=get_model(judge, reasoning_effort="high")
        )
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--auditor", choices=["anthropic/claude-opus-4-7", "anthropic/claude-opus-4-6"])
    parser.add_argument("-t", "--target")
    parser.add_argument("-j", "--judge", default="anthropic/claude-opus-4-7")
    parser.add_argument("-d", "--judge-dimensions", choices=["tags:safety", ], default=["tags: safety"])
    # parser.add_argument("-e", "--epoch", default=5) # trade epoch for longer tunrs 

    parser.add_argument("-t", default= 20, type=int)
    args = parser.parse_args()

    main(args.target, args.auditor, args.judge, args.judge_dimensions)