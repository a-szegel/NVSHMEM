import pytest

from utils import uid_init, mpi_init
from nvshmem.core import finalize

def pytest_addoption(parser):
    parser.addoption("--init-type", action="store", default="uid", help="Method to initialize NVSHMEM", choices=["uid", "mpi"])

@pytest.fixture(scope="session", autouse=True)
def nvshmem_init_fini(request):
    init_type = request.config.getoption("--init-type")
    if init_type == "uid":
        uid_init()
    elif init_type == "mpi":
        mpi_init()
    
    yield
    
    finalize()

