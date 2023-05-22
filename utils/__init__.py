import os
import logging
import platform
import logging.config
import torch
import torch.mps


class GPU:
    def __init__(self, is_init_device=False, empty_cache=False) -> None:
        self.gpu_allocation = {}
        self.is_cuda = torch.cuda.is_available()
        self.is_mps = torch.has_mps
        self.device_init = torch.device(f'mps') if self.is_mps else torch.device(f'cuda') if self.is_cuda else torch.device(f'cpu')
        if is_init_device and not self.is_cuda and not self.is_mps: torch.device(f'cpu')
        if self.is_mps:
            gpu_mem_alloc = torch.mps.current_allocated_memory() / (1024 ** 3)  # Convert bytes to gigabytes
            gpu_mem_cached = torch.mps.driver_allocated_memory() / (1024 ** 3) - gpu_mem_alloc  # Convert bytes to gigabytes
            self.gpu_allocation[f'GPU{0}'] = {'name': 'MPS', 'memory_allocated': f'{gpu_mem_alloc:.2f}G', 'memory_reserved': f'{gpu_mem_cached:.2f}G'}
        if self.is_cuda: 
            for i in range(torch.cuda.device_count()):
                cuda_device = torch.device(f'cuda:{i}')
                torch.cuda.set_device(cuda_device)
                gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to gigabytes
                gpu_mem_cached = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert bytes to gigabytes
                cuda_name = torch.cuda.get_device_name(cuda_device)
                self.gpu_allocation[f'GPU{i}'] = {'name': cuda_name, 'memory_allocated': f'{gpu_mem_alloc:.2f}G','memory_reserved': f'{gpu_mem_cached:.2f}G'}
        self.gpu_mem_description = f"{gpu_mem_alloc:.2f}G/{gpu_mem_cached:.2f}G" if self.is_cuda or self.is_mps else f"0.00G/0.00G"
        if empty_cache or is_init_device:
            if self.is_cuda: torch.cuda.empty_cache()
            if self.is_mps: torch.mps.empty_cache()

    def device(self) -> str:
        return self.device_init

GPU_DEVICE = GPU(is_init_device=True)
DEVICE = GPU_DEVICE.device()
LOGGING_NAME = 'CASUAL LLM'
VERBOSE = True
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans
EMOJI_SAFE_LOGGING = False

def colorstr(*input):
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')."""
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def emojis(string=''):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string

class EmojiFilter(logging.Filter):
    """
    A custom logging filter class for removing emojis in log messages.

    This filter is particularly useful for ensuring compatibility with Windows terminals
    that may not support the display of emojis in log messages.
    """

    def filter(self, record):
        """Filter logs by emoji unicode characters on windows."""
        record.msg = emojis(record.msg)
        return super().filter(record)

def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})

# Set logger
set_logging(LOGGING_NAME, verbose=VERBOSE)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if WINDOWS and EMOJI_SAFE_LOGGING:  # emoji-safe logging
    LOGGER.addFilter(EmojiFilter())

DEBUG_PREFIX = f"{colorstr('bright_yellow', 'DEBUG')} 🪲  - "


p_int_length = 40

def main_interface(conversation: str):
    # if WINDOWS: os.system("cls")
    # else: os.system("clear")
    print("="*p_int_length)
    print("Options:")
    print("1. Create Prompt")
    print("2. Display Text")
    print("3. Generate Text")
    print("4. Exit")
    print(conversation)
    print("="*p_int_length)