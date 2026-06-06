import dataclasses
import os
import sys

from ndsl import ndsl_log
from ndsl.optional_imports import cupy as cp


# Taken straight out of https://pcisig.com/membership/member-companies
_VENDOR_PCI_SIGNAURES = {
    0x10DE: "Nvidia",
    0x1002: "AMD",
    0x8086: "Intel",
    0x0: "Unknown",
}

# Cached copy of the hardware default
_GPU_HARDWARE_DEFAULTS = None


def _get_vendor() -> str:
    """Retrieve vendor using the current device PCI id to query the PCI vendor
    from the kernel logs

    ⚠️ Only works on Linux - kicks back to "Unknwon" in other cases
    """
    if not sys.platform.startswith("linux"):
        return _VENDOR_PCI_SIGNAURES[0x0]

    pci_device_id = cp.cuda.runtime.deviceGetPCIBusId(0)
    dev_path = f"/sys/bus/pci/devices/{pci_device_id}"
    if not os.path.exists(dev_path):
        return "Unknown"

    with open(os.path.join(dev_path, "vendor"), "r") as f:
        vendor_str = f.read().strip().replace("0x", "")
        vendor_id = int(vendor_str, 16)

    if vendor_id not in _VENDOR_PCI_SIGNAURES:
        ndsl_log.error(f"Unknown GPU vendor with PCI-SIG ID of {vendor_id:#X}")
        return "Unknown"
    return _VENDOR_PCI_SIGNAURES[int(vendor_str, 16)]


@dataclasses.dataclass
class GPUHardwareDefaults:
    """Compute defaults for common GPUs"""

    vendor: str
    block_size: list[int] = dataclasses.field(default_factory=list)
    compute_capability: int = -1  # Nvidia specific


def get_gpu_hardware_defaults() -> GPUHardwareDefaults:
    """Retrieve default values for GPU computation configuration"""
    global _GPU_HARDWARE_DEFAULTS
    if _GPU_HARDWARE_DEFAULTS is not None:
        return _GPU_HARDWARE_DEFAULTS  # type: ignore[unreachable]

    if not cp or not cp.cuda.is_available():
        ndsl_log.warning("No cupy - defaulting for GPU hardware")
        _GPU_HARDWARE_DEFAULTS = GPUHardwareDefaults(
            vendor="Unknown",
            block_size=[
                8,
                1,
                1,
            ],  # Smaller common denominator of massively parallel hardware
        )
        return _GPU_HARDWARE_DEFAULTS

    # Who goes there
    vendor = _get_vendor()
    if vendor == "Nvidia":
        compute_capability = int(cp.cuda.Device(0).compute_capability)
        # Default block size based on compute capability
        if compute_capability > 80:
            # Covers:
            #  - Blackwell (100+)
            #  - Hopper (90-100)
            #  - Ampere (80-90)
            block_sizes = [128, 1, 1]
        elif compute_capability > 60:
            # Covers:
            #  - Volta (70-80)
            #  - Pascal (60-70)
            block_sizes = [64, 8, 1]
        else:
            # For older hardware - we default to the safe warp-size since
            # the dawn of GPGPU on Nvidia hardware
            block_sizes = [32, 1, 1]

        _GPU_HARDWARE_DEFAULTS = GPUHardwareDefaults(
            vendor=vendor,
            block_size=block_sizes,
            compute_capability=compute_capability,
        )
    elif vendor == "AMD":
        _GPU_HARDWARE_DEFAULTS = GPUHardwareDefaults(
            vendor=vendor, block_size=[64, 1, 1]  # Default RDNA architectue is Wave64
        )
    elif vendor == "Intel":
        _GPU_HARDWARE_DEFAULTS = GPUHardwareDefaults(
            vendor=vendor,
            block_size=[32, 1, 1],  # Intel can run 8, 16 or 32 - but SIMD betters in 32
        )
    else:
        _GPU_HARDWARE_DEFAULTS = GPUHardwareDefaults(
            vendor=vendor,
            block_size=[
                8,
                1,
                1,
            ],  # Smaller common denominator of massively parallel hardware
        )

    ndsl_log.info(f"GPU vendor detected: {_GPU_HARDWARE_DEFAULTS.vendor}")

    return _GPU_HARDWARE_DEFAULTS
