"""Inspect available specializations; this command never launches a backend."""
import argparse
import json
from .catalog import describe, list_specializations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='List scopes without loading hardware dependencies')
    show = commands.add_parser('show', help='Show one specialization description')
    show.add_argument('specialization')
    args = parser.parse_args()
    try:
        result = (list_specializations() if args.command == 'list'
                  else describe(args.specialization))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
