import sys

def progress_spinner(text, total, current):
    spin_chars = ['-', '\\', '|', '/']
    idx = current % len(spin_chars)
    spinner = spin_chars[idx]
    count = f"{current}/{total}"
    sys.stdout.write(f"\r{text} {spinner} {count}")
    sys.stdout.flush()