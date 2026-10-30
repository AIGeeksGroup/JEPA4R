# Frozen native Test sources

The snapshot files retain their original bytes. Some logical native paths are backed by identical files elsewhere in this repository; `source_map.json` is the complete map. Restore the full original relative layout with the materializer rather than running an arbitrary file directly from the deduplicated snapshot directory.

See the repository-root `README_TEST_WORKFLOWS.md` for the visual comparison catalog and separate State baseline, actual recipes, external-asset requirements and validation boundaries.
