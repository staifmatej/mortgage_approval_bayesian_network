"""Module for interactive 'Press Enter to continue' functionality with animated dots."""
import sys
import select
import tty

def print_press_enter_to_continue():
    """Display animated 'Press Enter to continue' prompt with dot animation."""

    # Check if running in Jupyter/IPython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook
            input("\nPress Enter to continue...")
            return
    except NameError:
        pass  # Not in IPython/Jupyter
    try:
        import termios  # pylint: disable=import-outside-toplevel
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

        dots = ["   ", ".  ", ".. ", "..."]
        dot_index = 0

        print("\n")
        while 1:
            # Check if Enter was pressed
            if select.select([sys.stdin], [], [], 0.5)[0]:
                char = sys.stdin.read(1)
                if ord(char) == 13:  # Enter key
                    break

            # Update dots animation
            sys.stdout.write(f"\rPress Enter to continue{dots[dot_index]}")
            sys.stdout.flush()
            dot_index = (dot_index + 1) % len(dots)

        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\rPress Enter to continue...    ")

    except (ImportError, OSError, RuntimeError):
        input("\nPress Enter to continue...")
