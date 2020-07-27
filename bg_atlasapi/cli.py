from bg_atlasapi.list_atlases import show_atlases
from bg_atlasapi.update_atlases import update_atlas, install_atlas
from bg_atlasapi.config import cli_modify_config
import click


@click.command()
@click.argument("command")
@click.option("-s", "--show", is_flag=True)
@click.option("-a", "--atlas_name")
@click.option("-f", "--force", is_flag=True)
@click.option("-k", "--key")
@click.option("-v", "--value")
def bg_cli(
    command, atlas_name=None, force=False, show=False, key=None, value=None
):
    """
        Command line dispatcher. Given a command line call to `brainglobe`
        it calls the correct function, depending on which `command` was passed. 

        Arguments:
        ----------
        command: str. Name of the command:
            - list: list available atlases
            - install: isntall new atlas
            - update: update an installed atlas
            - config: modify config

        show: bool. If True when using `list` shows the local path of installed atlases
                and when using 'config' it prints the modify config results.
        atlas_name: ts. Used with `update` and `install`, name of the atlas to install
        force: bool, used with `update`. If True it forces the update
    """

    if command == "list":  # list atlases
        return show_atlases(show_local_path=show)

    elif command == "install":  # install atlas
        if atlas_name is None:
            raise ValueError(
                'No atlas named passed with command "install". Use the "-a"\
                                argument to pass an atls name'
            )
        return install_atlas(atlas_name=atlas_name)

    elif command == "update":  # update installed atlas
        if atlas_name is None:
            raise ValueError(
                'No atlas named passed with command "update". Use the "-a"\
                                argument to pass an atls name'
            )
        return update_atlas(atlas_name, force=force)

    elif command == "config":  # update config
        return cli_modify_config(key=key, value=value, show=show)

    else:  # command not recognized
        raise ValueError(
            f'Invalid command {command}. use "brainglobe -h for more info."'
        )
