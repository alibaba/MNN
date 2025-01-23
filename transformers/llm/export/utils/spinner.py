import time
import functools
import traceback
from yaspin import yaspin

RESET = "\033[0m"
GREEN = "\033[32;1m"
YELLOW = "\033[33;4m"

def spinner_run(text='Processing...', hide=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with yaspin(text=text, color="cyan") as spinner:
                start = time.time()
                try:
                    if hide: spinner.hide()
                    result = func(*args, **kwargs)
                    if hide: spinner.show()
                except Exception as e:
                    spinner.fail("💥 Failed")
                    traceback.print_exc()
                    exit(1)
                end = time.time()
                during = f'[{end-start:05.2f} s]'.replace('[0', '[ ')
                padding = ' ' * (64 - len(spinner.text) - len(result))
                spinner.text = f'{spinner.text}{YELLOW}{result}{RESET}{padding}{GREEN}{during}{RESET}'
                spinner.ok("✅ Done")
                return result
        return wrapper
    return decorator