import argparse
import re

class CSVorSpaceList(argparse.Action):
    """
    --arg a b,c d  -> ["a","b","c","d"]
    Comma/space mixed split + choices validation support
    """
    def __init__(self, option_strings, dest, nargs="+", choices=None, **kwargs):
        # Accept at least 1 by default; can change to '*' if needed
        if nargs is None:
            nargs = '+'
        self._choices = set(choices) if choices is not None else None
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not isinstance(values, (list, tuple)):
            values = [values]

        items = []
        for v in values:
            # Split by comma/space (ignore consecutive delimiters)
            parts = [s for s in re.split(r'[,\s]+', v) if s]
            items.extend(parts)

        # Validate choices
        if self._choices is not None:
            invalid = [x for x in items if x not in self._choices]
            if invalid:
                # argparse style error message
                msg = f"invalid choice(s): {invalid} (choose from {sorted(self._choices)})"
                raise argparse.ArgumentError(self, msg)

        setattr(namespace, self.dest, items)

# ----------------- Usage Example -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        action=CSVorSpaceList,
        choices=["train", "eval", "test"],   # Each element must be in this list
        help="Mode list separated by comma or space",
    )
    args = parser.parse_args()
    print(args.modes)
