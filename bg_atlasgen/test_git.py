from git import Repo
from pathlib import Path

GENERATION_DICT = dict(example_mouse=[100])


cwd = Path.home() / "bg_auto"
cwd.mkdir(exist_ok=True)


if __name__ == "__main__":
    repo_path = cwd / "atlas_repo"
    atlas_gen_path = Path(__file__).parent

    repo = Repo(repo_path)

    # repo.git.add(".")
    # repo.git.commit('-m', 'test commit', author='luigi.petrucco@gmail.com')
    repo.git.pull()
    repo.git.push()
