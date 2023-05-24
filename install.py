import sys
from utils import *

def install():
    # args
    args = sys.argv
    f_arg = None
    if len(args) > 1: f_arg = args[1]
    conda_env = None
    if len(args) > 2: conda_env = args[2]

    # stuff
    title = colorstr("bright_blue", "+ Casual LLM Installer")
    brks = ["{", "}"]

    # packages required
    # pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
    packages = ["transformers"]
    pip_only_packages = ["accelerate"]
    
    # error messages
    no_arg_error = colorstr("cyan", f'Use `conda {brks[0]}env_name{brks[1]}` or `pip` for argument\n')
    conda_env_error = colorstr("cyan", f"`python install.py conda {brks[0]}env_name{brks[1]}`")
    e_prefix = colorstr('bright_red', 'ERROR ❌️  ')

    # title
    LOGGER.info(f'{title}\n')

    # Installation with pip
    if f_arg == 'pip':
        try:
            # pip
            LOGGER.info(f'\t{colorstr("bright_cyan", "Installing with `pip`.")}\n')

            # package installations
            for package in packages:
                LOGGER.info(" ✅  " + colorstr("green", f"Installing {package}") + "\n")
                os.system(f'pip install {package}')
            for package in pip_only_packages:
                LOGGER.info(" ✅  " + colorstr("green", f"Installing {package}") + "\n")
                os.system(f'pip install {package}')
        except:
            # error
            LOGGER.error(f'{e_prefix}Package `{package}` failed to install using `pip`.\n', exc_info=1)
    
    # Installation with conda
    elif f_arg == 'conda':
        if conda_env is not None:
            os.system(f"conda activate {conda_env}")
            # conda with env
            LOGGER.info(colorstr("cyan", f"- Installing with `conda` in `{conda_env}` environment\n"))
            try:
                # package installations
                for package in packages:
                    LOGGER.info(" ✅  " + colorstr("green", f"Installing {package}") + "\n")
                    os.system(f"conda install -n {conda_env} {package}")
                for package in pip_only_packages:
                    LOGGER.info(" ✅  " + colorstr("green", f"Installing {package}") + "\n")
                    os.system(f'pip install {package}')
                LOGGER.info("\n- " + colorstr("cyan", f"✅  Done.\n"))
            except:
                # error
                LOGGER.error(f'{e_prefix}Package `{package}` failed to install for `{conda_env}` environment\n', exc_info=1)

        # no conda env
        else: LOGGER.error(f'{e_prefix}You are using conda without an environment specified.\n\n\tUse %s\n', conda_env_error)

    # no args
    else: LOGGER.error(f'\t{e_prefix}`arguments missing`\n\n\t{no_arg_error}')



# Installs dependencies.
# (more later on)
if __name__ == "__main__":
    install()
