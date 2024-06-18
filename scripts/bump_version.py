import argparse
import subprocess
import sys


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()


def run_towncrier():
    process = subprocess.Popen('towncrier', stdin=subprocess.PIPE, shell=True)
    process.communicate(input=b'N\n')


def process_command_line_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('increment', type=str, help='The increment type (major, minor, patch)')
    args = parser.parse_args()
    return args


def adjust_changelog():
    with open('CHANGELOG.rst', 'r') as file:
        lines = file.readlines()

    # Remove 'Elphick.' prefix from the first line
    prefix = 'Elphick.'
    if lines[0].startswith(prefix):
        lines[0] = lines[0][len(prefix):]

    # Adjust the length of the underline on the second line
    if lines[1].startswith('='):
        lines[1] = '=' * (len(lines[0].strip())) + '\n'  # -1 for the newline character

    with open('CHANGELOG.rst', 'w') as file:
        file.writelines(lines)


def main():
    args = process_command_line_parameters()

    increment = args.increment
    # Validate the input
    if increment not in ["major", "minor", "patch"]:
        print("Invalid version increment. Please enter 'major', 'minor', or 'patch'.")
        sys.exit(1)

    # Run the commands
    run_command(f"poetry version {increment}")
    run_command("poetry install --all-extras")

    run_towncrier()

    # remove the news fragments manually.
    run_command("rm -rf ./towncrier/newsfragments/*")

    # strip the Elphick. prefix from the top heading only.
    adjust_changelog()


if __name__ == "__main__":
    main()
