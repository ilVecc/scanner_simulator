import os
import sys
from contextlib import contextmanager

# TODO runs into OSError "Too many files open"
@contextmanager
def suppress_stdout(to=os.devnull):
    '''
    import os
    with suppress_stdout(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    stdout_old_fd = sys.stdout.fileno()  # this should be 1

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(stdout_new):
        sys.stdout.close() # + implicit flush()
        os.dup2(stdout_new.fileno(), stdout_old_fd) # 'old' writes to 'new' file
        sys.stdout = os.fdopen(stdout_old_fd, 'w') # Python writes to 'old'

    with os.fdopen(os.dup(stdout_old_fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(stdout_new=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(stdout_new=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different


# TODO THIS SHOULD ALSO WORK, BUT DOESN'T
# with open(os.devnull, 'w') as null, contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
#     << code here >>
