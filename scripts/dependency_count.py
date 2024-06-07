import subprocess
from collections import defaultdict

import toml
from pathlib import Path


def get_dependency_counts() -> str:
    # Load the pyproject.toml file
    pyproject = toml.load(Path(__file__).parents[1] / 'pyproject.toml')

    # Extract the extras dependencies
    extras_deps: set = {dep for extras in pyproject['tool']['poetry']['extras'].values() for dep in extras}
    base_deps: set = {dep for dep in pyproject['tool']['poetry']['dependencies'].keys() if dep not in extras_deps}

    def get_total_dep_count() -> int:
        command = ['poetry', 'show']
        result = subprocess.run(command, stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')

        total_packages = 0
        for line in output.split('\n'):
            if line and not line.startswith(' '):  # check if line is not empty and not indented
                total_packages += 1

        return total_packages

    def count_direct_child_dependencies():
        command = ['poetry', 'show', '--tree']
        result = subprocess.run(command, stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')

        dependencies = defaultdict(int)
        current_dependency = None

        for line in output.split('\n'):
            stripped_line = line.strip()
            if stripped_line:  # check if line is not empty
                if not line.startswith(' '):  # check if line is not indented
                    if not stripped_line.startswith(('├──', '└──', '│')):  # check if line is a primary dependency
                        current_dependency = stripped_line.split(' ')[
                            0]  # only consider the first word as the name of the dependency
                        dependencies[current_dependency] = 0
                elif current_dependency is not None and (line.startswith('    ├──') or line.startswith(
                        '    └──')):  # check if line starts with four spaces and either '├──' or '└──'
                    dependencies[current_dependency] += 1
                elif current_dependency is not None and not (line.startswith('    ├──') or line.startswith('    └──')):
                    current_dependency = None  # reset current_dependency if line is indented with four spaces but does not start with either '├──' or '└──'

        return dependencies

    res: str = ""
    # Count base dependencies
    all_dep_counts: dict = count_direct_child_dependencies()
    base_dep_counts: dict = {dep: count for dep, count in all_dep_counts.items() if
                             dep in base_deps}
    res += f'Base dependencies: {len(base_dep_counts.keys()) + sum(base_dep_counts.values())}'
    for dep, count in base_dep_counts.items():
        res += f'\n{dep}: {count} direct child dependencies'

    # Count dev dependencies
    dev_dep_counts = {dep: count for dep, count in all_dep_counts.items() if dep not in base_deps.union(extras_deps)}
    res += f'\n\nDev dependencies: {len(dev_dep_counts.keys()) + sum(dev_dep_counts.values())}'
    for dep, count in dev_dep_counts.items():
        res += f'\n{dep}: {count} direct child dependencies'

    # Count extras dependencies
    ext_dep_counts = {dep: count for dep, count in all_dep_counts.items() if dep in extras_deps}
    res += f'\n\nExtras dependencies: {len(ext_dep_counts.keys()) + sum(ext_dep_counts.values())}'
    for dep, count in ext_dep_counts.items():
        res += f'\n{dep}: {count} direct child dependencies'

    res += f'\n\nTotal dependencies: {str(get_total_dep_count())}\n'

    return res


print(get_dependency_counts())
