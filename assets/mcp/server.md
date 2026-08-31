A git-backed markdown corpus indexed for retrieval — not a filesystem to browse
directly. Document *text* matches terms, not substrings — there is no regex. Paths
are the exception: `path_prefix` is a case-insensitive substring match, so a
half-remembered filename is enough to find a document.
