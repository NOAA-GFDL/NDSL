import ndsl.stencils.testing.conftest
from ndsl.stencils.testing.conftest import *  # noqa: F403,F401

import savepoint

# Point to an __init__.py where all the TestX are improted
ndsl.stencils.testing.conftest.translate = savepoint  # type: ignore
