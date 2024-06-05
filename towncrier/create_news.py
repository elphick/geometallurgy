import os
import subprocess
from pathlib import Path


def create_news_fragments():
    # Get the commit hashes and messages from the current branch
    result = subprocess.run(['git', 'log', '--pretty=format:%h %s'], stdout=subprocess.PIPE)
    commits = result.stdout.decode('utf-8').split('\n')

    for commit in commits:
        hash, message = commit.split(' ', 1)

        # Create a news fragment file for each commit
        filename = Path(f'newsfragments/{hash}.bugfix')
        with open(filename, 'w') as f:
            f.write(message)

        print(f'Created file: {filename.name}')


if __name__ == '__main__':
    create_news_fragments()
