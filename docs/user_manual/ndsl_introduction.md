<style>
/* re-enable the left side navigation bar for this page */
@media screen and (min-width: 76.1875em) {
  .md-sidebar--primary {
    display: block !important;
  }
}
</style>

# NDSL User Manual

The NASA–NOAA Domain-Specific Language (NDSL) is a modern framework for developing high-performance Earth system model components with a focus on portability, readability, and computational efficiency. NDSL enables scientists and developers to write expressive, maintainable code that can target multiple hardware architectures, including CPUs and GPUs, while preserving scientific integrity and performance.

NDSL is designed to bridge the gap between atmospheric science and modern software engineering practices. By abstracting backend-specific implementation details, NDSL allows users to focus on scientific development while leveraging scalable and portable computational infrastructure underneath.

This documentation serves as both an introduction for new users and a reference for active developers working within the NDSL framework.


### What You Will Learn
The NDSL user guide introduces relevant concepts and provides knowledge needed to use NDSL. By the end of this guide, users will be introduced to the following:

- [Data types and storage objects](./user_manual/data.md)
- [Writing basic NDSL code](./user_manual/writing_ndsl_code.md)
- [Advanced features and capabilities](./user_manual/advanced_features.md)
- [Common patterns](./user_manual/common_patterns.md)
- [Best coding practices](./user_manual/best_coding_practices.md)
- [How and why classes are used with NDSL](./user_manual/why_use_classes.md)
- [Underlying NDSL infrastructure](./user_manual/backends.md)

### Intended Audience

This guide is intended for:

- Atmospheric and Earth system scientists
- Developers porting legacy model components
- Researchers exploring CPU/GPU portability
- Contributors developing new NDSL features and workflows

Whether you are getting started with NDSL concepts or actively integrating NDSL into existing modeling systems, this documentation is designed to provide the foundational knowledge needed to work effectively within the framework.
